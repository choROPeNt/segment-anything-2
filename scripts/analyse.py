from __future__ import annotations

import gc
import os
import random
import sys
import argparse
import warnings
from typing import cast, Any, Tuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import distance_transform_edt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # suppress DecompressionBombWarning for large microscopy images
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.addons import (Sam2Patcher, 
                         show_anns, 
                         write_h5, 
                         delete_h5_if_exists)

seed = 67
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


 # move results fully off GPU just in case
def to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def make_mask_generator(sam2):
    return SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=48,
        points_per_batch=164,          # lower than 96 to reduce peak memory
        pred_iou_thresh=0.1,
        min_mask_region_area=150,
        box_nms_thresh=0.1,
        stability_score_thresh=0.9,
        # multimask_output=False,       # important
        # output_mode="uncompressed_rle"  # important
    )





def select_device():
    """
    Select computation device (CUDA / MPS / CPU)
    and apply safe backend settings.
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")

    if device.type == "cuda":
        # Optional: enable TF32 where supported, but only if the attributes exist
        try:
            if torch.cuda.get_device_properties(0).major >= 8:
                if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True
        except Exception as e:
            print(f"Warning: could not set TF32 flags: {e}")

    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "\ngive numerically different outputs and sometimes degraded performance on MPS. "
            "\nSee e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    return device

def load_image_from_path(file_path):
    """
    Load multiple images (grayscale or RGB) into NumPy arrays.

    Args:
        file_paths (list[str]): List of full image file paths.

    Returns:
        list[dict]: Each entry has keys:
                    'name' (str), 'array' (np.ndarray), 'grayscale' (bool)
    """

    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    image = Image.open(file_path)
    was_grayscale = False

    if image.mode in ("L", "I"):  # grayscale
        was_grayscale = True
        image = image.convert("RGB")
    elif image.mode == "RGBA":    # drop alpha
        image = image.convert("RGB")

    image_np = np.asarray(image)
    print(f"{file_name:20s} | shape={image_np.shape}, \
            dtype={image_np.dtype}, \
            grayscale={was_grayscale}" 
            )


    return image_np, was_grayscale


def find_mask_path(image_path):
    """
    Look for a companion mask next to image_path, following the
    '<stem>_mask.<ext>' convention (e.g. img.tiff -> img_mask.png).

    Returns:
        str | None: path to the mask file, or None if not found.
    """
    folder = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = os.path.join(folder, f"{stem}_mask{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def load_mask_from_path(mask_path, target_shape):
    """
    Load a mask image and binarize it to {0,1}.

    Args:
        mask_path (str): path to the mask file.
        target_shape (tuple[int, int]): expected (H, W), must match the image.

    Returns:
        np.ndarray: binary mask of shape target_shape, dtype uint8.
    """
    mask = np.asarray(Image.open(mask_path).convert("L"))

    if mask.shape != tuple(target_shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {target_shape} for '{mask_path}'"
        )

    return (mask > 0).astype(np.uint8)


def extract_mask_aware_patches(
    image,                   # 2-D array
    mask,                    # 2-D binary array {0,1}
    patch_size   = 256,
    stride       = 128,      # used when refine=False
    min_cov      = 0.5,
    refine       = True,     # False → fast path: grid + coverage filter only
    stride_min   = 64,       # refine=True only
    stride_max   = 256,      # refine=True only
    boundary_sharpness = 4.0,
    max_patches  = None,
):
    """
    Extract mask-aware patches from *image* guided by *mask*.

    refine=False (fast path)
    ------------------------
    Regular grid with *stride*, keep every patch whose mask-coverage >= min_cov.
    Coverage is computed in one vectorised pass with sliding_window_view — no loops.

    refine=True (slow path)
    -----------------------
    Dense candidate grid (stride_min), greedy NMS with variable suppression radius
    derived from distance-to-boundary transform.

    Returns
    -------
    coords_kept   : list of (y, x) top-left corners
    img_patches   : float32 tensor  [N, p, p]
    mask_patches  : float32 tensor  [N, p, p]
    coverage      : float32 tensor  [N]
    """
    H, W = mask.shape
    p    = patch_size

    if H < p or W < p:
        raise ValueError("Image smaller than patch_size.")

    mask_f = mask.astype(np.float32)
    img_f  = image.astype(np.float32)

    # ------------------------------------------------------------------ #
    # FAST PATH                                                            #
    # ------------------------------------------------------------------ #
    if not refine:
        # sliding_window_view gives a zero-copy view: (ny, nx, p, p)
        mask_wins = sliding_window_view(mask_f, (p, p))[::stride, ::stride]
        img_wins  = sliding_window_view(img_f,  (p, p))[::stride, ::stride]

        ny, nx = mask_wins.shape[:2]
        cov = mask_wins.sum(axis=(-1, -2)) / float(p * p)  # (ny, nx)

        ys = np.arange(0, H - p + 1, stride)
        xs = np.arange(0, W - p + 1, stride)

        keep_y, keep_x = np.where(cov >= min_cov)

        if max_patches is not None and len(keep_y) > max_patches:
            # sort by coverage descending, take top-N
            top = np.argsort(cov[keep_y, keep_x])[::-1][:max_patches]
            keep_y, keep_x = keep_y[top], keep_x[top]

        coords_kept  = [(int(ys[iy]), int(xs[ix])) for iy, ix in zip(keep_y, keep_x)]
        img_patches  = torch.from_numpy(img_wins [keep_y, keep_x].copy())
        mask_patches = torch.from_numpy(mask_wins[keep_y, keep_x].copy())
        coverage     = torch.from_numpy(cov      [keep_y, keep_x].astype(np.float32).copy())

        return coords_kept, img_patches, mask_patches, coverage

    # ------------------------------------------------------------------ #
    # REFINEMENT PATH (greedy NMS with variable suppression radius)        #
    # ------------------------------------------------------------------ #
    mask_wins = sliding_window_view(mask_f, (p, p))[::stride_min, ::stride_min]
    img_wins  = sliding_window_view(img_f,  (p, p))[::stride_min, ::stride_min]

    ny, nx   = mask_wins.shape[:2]
    cov_grid = mask_wins.sum(axis=(-1, -2)) / float(p * p)

    ys = np.arange(0, H - p + 1, stride_min)
    xs = np.arange(0, W - p + 1, stride_min)

    iy_all, ix_all = np.where(cov_grid >= min_cov)
    if iy_all.size == 0:
        return [], torch.empty(0, p, p), torch.empty(0, p, p), torch.empty(0)

    cov_keep = cov_grid[iy_all, ix_all]

    # sort by coverage descending for greedy selection
    order    = np.argsort(cov_keep)[::-1]
    iy_all   = iy_all[order];  ix_all = ix_all[order];  cov_keep = cov_keep[order]

    # variable suppression radius from distance transform
    dist     = distance_transform_edt(mask.astype(bool))
    dist_norm = np.tanh(dist / max(1.0, p / boundary_sharpness))
    rad_map  = stride_min + (stride_max - stride_min) * dist_norm

    # patch centres
    cy_all = ys[iy_all] + p / 2.0
    cx_all = xs[ix_all] + p / 2.0

    taken       = np.zeros(len(iy_all), dtype=bool)
    kept_indices = []

    for i in range(len(iy_all)):
        if taken[i]:
            continue
        kept_indices.append(i)
        taken[i] = True

        ri = float(rad_map[
            int(np.clip(round(cy_all[i]), 0, H-1)),
            int(np.clip(round(cx_all[i]), 0, W-1)),
        ])
        dy = cy_all[i+1:] - cy_all[i]
        dx = cx_all[i+1:] - cx_all[i]
        taken[i+1:] |= (dy*dy + dx*dx) < ri*ri

        if max_patches is not None and len(kept_indices) >= max_patches:
            break

    ki = np.array(kept_indices)
    coords_kept  = [(int(ys[iy_all[k]]), int(xs[ix_all[k]])) for k in ki]
    img_patches  = torch.from_numpy(img_wins [iy_all[ki], ix_all[ki]].copy())
    mask_patches = torch.from_numpy(mask_wins[iy_all[ki], ix_all[ki]].copy())
    coverage     = torch.from_numpy(cov_keep [ki].astype(np.float32).copy())

    return coords_kept, img_patches, mask_patches, coverage


#------------------
#------ MAIN ------
#------------------
def main(args,
        valid_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ):
    
    # Parse arguments
    input_path = args.input
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)
    print(f"Analyzing data in {input_path}, results will be saved to {output_path}")

    # get list of valid files in input path
    if args.dir_flag:
        files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
        file_paths = [os.path.join(input_path, f) for f in files]
        file_names  = [os.path.splitext(f)[0] for f in files]
        print(f"Found {len(files)} valid files in {input_path}")
    else:
        if input_path.lower().endswith(valid_exts):
            files = [os.path.basename(input_path)]
            input_path = os.path.dirname(input_path)
            file_paths = [os.path.join(input_path, f) for f in files]
            file_names  = [os.path.splitext(f)[0] for f in files]
            print(f"Processing single file: {files[0]}")
        else:
            print(f"Error: input file '{input_path}' does not have a valid image extension.")
            sys.exit(1)

    #------------------------
    # Set-Up Model SAM2
    #------------------------
    device = select_device()

    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=str(device), apply_postprocessing=True)


    
    ## intialize tiler 
    patcher = Sam2Patcher(
        patch_h=512, 
        patch_w=512, 
        overlap_h=64, 
        overlap_w=64, 
        pad_mode="constant",
        merge_iou=0.1
    )




    for i, (file,file_name, file_path) in enumerate(zip(files,file_names,file_paths)):
        print("#----------------------------")
        print(f"Processing file {i+1}/{len(files)}: {file}")
        image, was_grayscale  = load_image_from_path(file_path)

        # image = image[:,:2048]
        ## Patching
        if args.mask_aware:
            mask_path = find_mask_path(file_path)
            if mask_path is None:
                raise FileNotFoundError(
                    f"--mask_aware is set but no mask found for '{file_path}' "
                    f"(expected '{os.path.splitext(file_path)[0]}_mask.<ext>')"
                )
            mask = load_mask_from_path(mask_path, target_shape=image.shape[:2])
            print(f"Using mask-aware tiling with mask: {os.path.basename(mask_path)}")

            coords_kept, _, _, coverage = extract_mask_aware_patches(
                image=mask,
                mask=mask,
                patch_size=args.mask_patch_size,
                stride=args.mask_stride,
                min_cov=args.mask_min_cov,
                refine=args.mask_refine,
                max_patches=args.mask_max_patches,
            )

            if not coords_kept:
                print(f"No patches met min_cov={args.mask_min_cov} for {file}, skipping.")
                continue

            p = args.mask_patch_size
            patches = [image[y:y + p, x:x + p, ...].copy() for y, x in coords_kept]
            offsets = coords_kept
            padded_shape = image.shape

            print(f"Mask-aware tiling kept {len(patches)} patches "
                  f"(avg coverage {float(coverage.mean()):.2f}) of shape {patches[0].shape}")
        else:
            patches, offsets, padded_shape = patcher.tile_numpy(image)
            print(f"Tiled into {len(patches)} patches of shape {patches[0].shape}, \
                  padded shape: {padded_shape}")

        # Pre-Allocate some Space
        h5_path = os.path.join(output_path,f"{file_name}.patches.result.h5")

        delete_h5_if_exists(h5_path)
        all_detections = []

        mask_generator = make_mask_generator(sam2)

        for patch_idx, p in enumerate(tqdm(patches, total=len(patches), desc="Mask generation", unit="patch")):

            if device.type == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    detections = mask_generator.generate(p)
            else:
                with torch.inference_mode():
                    detections = mask_generator.generate(p)

            mask_generator.predictor.reset_predictor()


            filtered = [
                d for d in detections
                if (args.min_area is None or d["area"] >= args.min_area)
                and (args.max_area is None or d["area"] < args.max_area)
            ]
            all_detections.append(filtered)

            del detections
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    
        print(cast(Tuple[int, int], padded_shape[:2]))

        label_map, instances = patcher.stitch_sam2_instances(
            padded_shape_hw=cast(Tuple[int, int], padded_shape[:2]),
            offsets=offsets,
            per_tile_sam2=all_detections,
            debug=False,
        )

        if args.save_fig:
            plt.figure()
            plt.imshow(image)
            show_anns(instances)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path,f"{file_name}_results.png"),dpi=300)


        if was_grayscale and image.ndim == 3:
            # test if all channels are equal (typical for grayscale expanded to RGB)
            if np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2]):
                image = image[..., 0]  # take one channel

        dict_out = {
            "image": image ,
            "labels": label_map,
            "instances": instances
        }

        write_h5(path = os.path.join(output_path,file_name + ".h5"), dict_out= dict_out)


#------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input dataset dir or file to be analyzed.'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=None,
        help='Optional path to save the analysis results (e.g. HDF5, or image outputs).'
    )

    parser.add_argument(
        '--min_area',
        type=float,
        required=False,
        default=None,
        help='Optional minimum area threshold; detections with area < min_area are excluded.'
    )

    parser.add_argument(
        '--max_area',
        type=float,
        required=False,
        default=None,
        help='Optional maximum area threshold; detections with area >= max_area are excluded.'
    )

    parser.add_argument(
        '--save_fig',
        action='store_true',
        help='If set, saves generated figures instead of only displaying them.'
    )

    parser.add_argument(
        '--mask_aware',
        action='store_true',
        help="If set, use mask-guided patch extraction instead of a regular grid. "
             "Requires a '<stem>_mask.<ext>' file next to each image; errors out if missing."
    )

    parser.add_argument(
        '--mask_patch_size',
        type=int,
        default=256,
        help='Patch size (square) used for mask-aware tiling.'
    )

    parser.add_argument(
        '--mask_stride',
        type=int,
        default=128,
        help='Grid stride for mask-aware tiling (fast path, i.e. --mask_refine not set).'
    )

    parser.add_argument(
        '--mask_min_cov',
        type=float,
        default=0.5,
        help='Minimum mask coverage fraction required to keep a patch.'
    )

    parser.add_argument(
        '--mask_refine',
        action='store_true',
        help='Use the slower greedy-NMS refinement pass instead of a fixed grid for mask-aware tiling.'
    )

    parser.add_argument(
        '--mask_max_patches',
        type=int,
        default=None,
        help='Optional cap on the number of mask-aware patches kept per image.'
    )


    args = parser.parse_args()
    
    # Check that input is a directory
    if os.path.isdir(args.input):
        args.dir_flag = True
        print("Input is a directory")
    else:
        args.dir_flag = False
        print("Input is a single file ")


    # If output not given → create results folder in input path
    if args.output is None:
        if args.dir_flag:
            args.output = os.path.join(args.input, "results")
        else:
            args.output = os.path.join(os.path.dirname(args.input), "results")
        
        os.makedirs(args.output, exist_ok=True)

    main(args)