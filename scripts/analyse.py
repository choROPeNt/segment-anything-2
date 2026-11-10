from __future__ import annotations

import gc
import os
import random
import sys
import argparse
import warnings
from typing import cast, Any, Tuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")

import h5py
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
                         generate_upscaled,
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


BASE_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
# fine-tuned Hiera-T (fibre_ds_512_pad256, 60 epochs); default AMG filters below from the sweep in
# <run>/eval_val_opti (scripts/sweep_amg.py, 16 points per side: F1 1.000 vs. 0.991 with the old filters)
DEFAULT_CHECKPOINT = ("sam2_logs/configs/sam2.1_training/sam2.1_hiera_t_fibre_ds_512_pad256_train-val_diag_e60.yaml/"
                      "checkpoints/checkpoint_060.pt")


def make_mask_generator(sam2, points_per_side=48, pred_iou_thresh=0.1, stability_score_thresh=0.9,
                        box_nms_thresh=0.1, min_mask_region_area=150):
    return SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=64,           # library default; 164 was above default and drove peak memory up
        pred_iou_thresh=pred_iou_thresh,
        min_mask_region_area=min_mask_region_area,
        box_nms_thresh=box_nms_thresh,
        stability_score_thresh=stability_score_thresh,
        multimask_output=False,        # 1 mask/point instead of 3 -> ~3x less decoder memory
        # output_mode="uncompressed_rle"  # important
    )





def amg_params(args):
    """AMG settings of this run (also stored as attributes of the result .h5)."""
    return {
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "min_mask_region_area": args.min_mask_region_area,
    }


def output_tag(args):
    """
    Suffix of the output names: --tag if given, else none for the base model and the checkpoint
    file stem (e.g. ``checkpoint_060``) for any other, so results of different checkpoints do not
    overwrite each other.
    """
    if args.tag is not None:
        return args.tag
    if os.path.normpath(args.checkpoint) == os.path.normpath(BASE_CHECKPOINT):
        return ""
    return os.path.splitext(os.path.basename(args.checkpoint))[0]


def shape_metrics(seg):
    """
    Roundness of one binary instance mask.

    Returns:
        circularity: 4*pi*A / P^2 (1 = disk, lower for ragged / concave / fragmented shapes). P is the
                     summed outer contour length of all components, so split instances score low. Clipped
                     to 1 (pixel contours slightly underestimate P of small disks).
        axis_ratio:  minor / major axis of the moment-equivalent ellipse (1 = round, -> 0 = elongated).
    """
    import cv2

    seg = np.asarray(seg, dtype=bool)
    area = int(seg.sum())
    if area < 3:
        return 0.0, 0.0

    padded = np.pad(seg, 1).astype(np.uint8)          # contours touching the crop edge stay closed
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    circularity = min(1.0, 4.0 * np.pi * area / perimeter**2) if perimeter > 0 else 0.0

    ys, xs = np.nonzero(seg)
    cov = np.cov(np.stack([xs, ys]).astype(np.float64))
    ev = np.linalg.eigvalsh(cov)                      # ascending
    axis_ratio = float(np.sqrt(max(ev[0], 0.0) / ev[1])) if ev[1] > 0 else 0.0

    return float(circularity), axis_ratio


def filter_round_instances(label_map, instances, min_circularity=0.0, min_axis_ratio=0.0):
    """
    Drop stitched instances that are not nearly round: circularity < min_circularity or
    axis_ratio < min_axis_ratio (0 disables a criterion). Removed instances are set to 0 in
    *label_map* (in place); kept instances get 'circularity' / 'axis_ratio' entries.
    """
    if min_circularity <= 0 and min_axis_ratio <= 0:
        return label_map, instances

    kept = []
    for inst in instances:
        circ, ar = shape_metrics(inst["segmentation_crop"])
        if circ >= min_circularity and ar >= min_axis_ratio:
            inst["circularity"] = circ
            inst["axis_ratio"] = ar
            kept.append(inst)
        else:
            x, y, w, h = inst["bbox"]
            sub = label_map[y:y + h, x:x + w]
            sub[sub == inst["id"]] = 0

    print(f"Roundness filter (circularity >= {min_circularity}, axis ratio >= {min_axis_ratio}): "
          f"kept {len(kept)}/{len(instances)} instances")
    return label_map, kept


def filter_instances_in_mask(label_map, instances, mask, min_inside=0.5):
    """
    Drop stitched instances lying outside the region *mask*: an instance is kept only if at least
    *min_inside* of its pixels fall inside the mask (1.0 = completely inside). Removed instances
    are set to 0 in *label_map* (in place); kept instances get a 'mask_inside' entry.
    """
    kept = []
    for inst in instances:
        x, y, w, h = inst["bbox"]
        seg = np.asarray(inst["segmentation_crop"], dtype=bool)
        inside = float(mask[y:y + h, x:x + w][seg].sum()) / max(int(seg.sum()), 1)
        if inside >= min_inside:
            inst["mask_inside"] = inside
            kept.append(inst)
        else:
            sub = label_map[y:y + h, x:x + w]
            sub[sub == inst["id"]] = 0

    print(f"Mask filter (>= {min_inside:.0%} of the object inside the mask): "
          f"kept {len(kept)}/{len(instances)} instances")
    return label_map, kept


def otsu_threshold(values, nbins=256):
    """Otsu threshold of a 1-D array (any dtype): the value best separating dark and bright pixels."""
    values = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)              # pixels at or below each bin
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2                 # between-class variance
    return float(edges[int(np.argmax(between[:-1])) + 1])  # upper edge of the last dark bin


def filter_bright_instances(label_map, instances, image, region=None, min_bright_fraction=0.5,
                            threshold=None):
    """
    Drop stitched instances that are not bright: fibres are bright, the matrix between them is dark.
    An instance is kept if at least *min_bright_fraction* of its pixels are brighter than *threshold*
    (default: Otsu threshold of the image, computed only inside *region* (the mask) if given).
    Removed instances are set to 0 in *label_map* (in place); kept instances get 'mean_intensity'
    and 'bright_fraction' entries.
    """
    gray = image.mean(axis=-1) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.float32)
    if threshold is None:
        sample = gray[region.astype(bool)] if region is not None else gray
        threshold = otsu_threshold(sample)
        how = "Otsu" + (" inside mask" if region is not None else "")
    else:
        how = "given"

    kept = []
    for inst in instances:
        x, y, w, h = inst["bbox"]
        seg = np.asarray(inst["segmentation_crop"], dtype=bool)
        vals = gray[y:y + h, x:x + w][seg]
        bright = float((vals > threshold).mean()) if vals.size else 0.0
        if bright >= min_bright_fraction:
            inst["mean_intensity"] = float(vals.mean())
            inst["bright_fraction"] = bright
            kept.append(inst)
        else:
            sub = label_map[y:y + h, x:x + w]
            sub[sub == inst["id"]] = 0

    print(f"Brightness filter (>= {min_bright_fraction:.0%} of pixels above {threshold:.1f}, {how}): "
          f"kept {len(kept)}/{len(instances)} instances")
    return label_map, kept, threshold


def load_sam2(model_cfg, checkpoint, device):
    """build_sam2 with a readable error for a missing checkpoint or a checkpoint / config mismatch."""
    if not os.path.isfile(checkpoint):
        sys.exit(f"Error: checkpoint '{checkpoint}' not found.")
    try:
        return build_sam2(model_cfg, checkpoint, device=str(device), apply_postprocessing=True)
    except RuntimeError as e:
        sys.exit(f"Error: could not load '{checkpoint}' with model config '{model_cfg}' ({e}).\n"
                 "The --config must match the model size the checkpoint was trained with, e.g. "
                 "configs/sam2.1/sam2.1_hiera_t.yaml for Hiera-T fine-tunes (not the training config).")


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
    '<stem>.mask.<ext>' convention (e.g. img.tiff -> img.mask.png).

    Returns:
        str | None: path to the mask file, or None if not found.
    """
    folder = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        candidate = os.path.join(folder, f"{stem}.mask{ext}")
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
    patch_size   = 512,
    stride       = 256,      # used when refine=False
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
    dist     = np.asarray(distance_transform_edt(mask.astype(bool)))   # array (typed as tuple | None by scipy)
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
        files = sorted(f for f in os.listdir(input_path)
                       if f.lower().endswith(valid_exts)
                       and not os.path.splitext(f)[0].endswith(".mask"))   # skip companion masks
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

    sam2_checkpoint = args.checkpoint
    model_cfg = args.config
    tag = output_tag(args)
    suffix = f".{tag}" if tag else ""
    print(f"checkpoint: {sam2_checkpoint}  (config {model_cfg})")
    print(f"output names: <file>{suffix}.h5" + (f"  (tag '{tag}', set with --tag)" if tag else ""))

    sam2 = load_sam2(model_cfg, sam2_checkpoint, device)


    
    ## intialize tiler 
    patcher = Sam2Patcher(
        patch_h=args.patch_size,
        patch_w=args.patch_size,
        overlap_h=args.overlap,
        overlap_w=args.overlap,
        pad_mode="constant",
        merge_iou=args.merge_iou
    )




    for i, (file,file_name, file_path) in enumerate(zip(files,file_names,file_paths)):
        print("#----------------------------")
        print(f"Processing file {i+1}/{len(files)}: {file}")
        image, was_grayscale  = load_image_from_path(file_path)

        # image = image[:,:2048]
        ## Patching
        if args.mask_aware:
            mask_path = args.mask if args.mask else find_mask_path(file_path)
            if mask_path is None:
                raise FileNotFoundError(
                    f"--mask_aware is set but no mask found for '{file_path}' "
                    f"(expected '{os.path.splitext(file_path)[0]}.mask.<ext>', or pass --mask)"
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

        mask_generator = make_mask_generator(sam2, **amg_params(args))
        if args.upscale_to:
            print(f"Patches are upscaled to {args.upscale_to} px for SAM 2 and the masks scaled back")

        for patch_idx, p in enumerate(tqdm(patches, total=len(patches), desc="Mask generation", unit="patch")):

            if device.type == "cuda":
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    detections = generate_upscaled(mask_generator, p, args.upscale_to)
            else:
                with torch.inference_mode():
                    detections = generate_upscaled(mask_generator, p, args.upscale_to)

            mask_generator.predictor.reset_predictor()


            filtered = [
                d for d in detections
                if (args.min_area is None or d["area"] >= args.min_area)
                and (args.max_area is None or d["area"] < args.max_area)
            ]
            all_detections.append(filtered)

            del detections
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    
        print(cast(Tuple[int, int], padded_shape[:2]))

        label_map, instances = patcher.stitch_sam2_instances(
            padded_shape_hw=cast(Tuple[int, int], padded_shape[:2]),
            offsets=offsets,
            per_tile_sam2=all_detections,
            debug=False,
        )

        if args.mask_aware:
            label_map, instances = filter_instances_in_mask(
                label_map, instances, mask, min_inside=args.mask_min_inside,
            )

        bright_threshold = args.bright_threshold
        if args.min_bright_fraction > 0:
            label_map, instances, bright_threshold = filter_bright_instances(
                label_map, instances, image,
                region=mask if args.mask_aware else None,
                min_bright_fraction=args.min_bright_fraction,
                threshold=args.bright_threshold,
            )

        label_map, instances = filter_round_instances(
            label_map, instances,
            min_circularity=args.min_circularity,
            min_axis_ratio=args.min_axis_ratio,
        )

        if args.save_fig:
            plt.figure()
            plt.imshow(image)
            show_anns(instances)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path,f"{file_name}{suffix}_results.png"),dpi=300)


        if was_grayscale and image.ndim == 3:
            # test if all channels are equal (typical for grayscale expanded to RGB)
            if np.allclose(image[..., 0], image[..., 1]) and np.allclose(image[..., 1], image[..., 2]):
                image = image[..., 0]  # take one channel

        dict_out = {
            "image": image ,
            "labels": label_map,
            "instances": instances
        }

        out_h5 = os.path.join(output_path, f"{file_name}{suffix}.h5")
        write_h5(path=out_h5, dict_out=dict_out)
        with h5py.File(out_h5, "a") as hf:       # provenance: which model / settings made these masks
            hf.attrs["sam2_checkpoint"] = os.path.abspath(sam2_checkpoint)
            hf.attrs["sam2_config"] = model_cfg
            hf.attrs["sam2_tag"] = tag
            for k, v in amg_params(args).items():
                hf.attrs[f"amg_{k}"] = v
            hf.attrs["min_circularity"] = args.min_circularity
            hf.attrs["min_axis_ratio"] = args.min_axis_ratio
            hf.attrs["merge_iou"] = args.merge_iou
            hf.attrs["min_bright_fraction"] = args.min_bright_fraction
            if bright_threshold is not None:
                hf.attrs["bright_threshold"] = bright_threshold
            if args.mask_aware:
                hf.attrs["mask_path"] = os.path.abspath(mask_path)
                hf.attrs["mask_min_inside"] = args.mask_min_inside
        print(f"saved {out_h5}")


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
        default="results",
        help='Optional path to save the analysis results (e.g. HDF5, or image outputs).'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=DEFAULT_CHECKPOINT,
        help='SAM 2 checkpoint (default: the fine-tuned e60 checkpoint; base model: '
             f'{BASE_CHECKPOINT}).'
    )

    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Suffix of the output names (<file>.<tag>.h5). Default: none for the base model, '
             'the checkpoint file stem otherwise (checkpoint_060). Pass --tag "" to write plain <file>.h5.'
    )

    parser.add_argument(
        '--config',
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help='Model config matching the checkpoint (model size; for a Hiera-T fine-tune the same '
             'configs/sam2.1/sam2.1_hiera_t.yaml, not the training config).'
    )

    parser.add_argument(
        '--patch_size',
        type=int,
        default=512,
        help='Patch size of the regular tiling (e.g. 256 for 20x images).'
    )

    parser.add_argument(
        '--merge_iou',
        type=float,
        default=0.1,
        help='Stitching: a detection is merged into an existing instance of a neighbouring patch '
             'if their IoU is at least this, otherwise it becomes a new instance. Higher = fewer '
             'merges (more split / duplicate fibres at patch borders), lower = more merges '
             '(touching fibres may fuse). Default 0.1.'
    )

    parser.add_argument(
        '--overlap',
        type=int,
        default=64,
        help='Overlap between neighbouring patches in px.'
    )

    parser.add_argument(
        '--upscale_to',
        type=int,
        default=0,
        help='Upscale every patch to this size (longer side) before SAM 2, masks are scaled back '
             '(e.g. 1024 for 256 px patches = factor 4). 0 = off.'
    )

    parser.add_argument(
        '--points_per_side',
        type=int,
        default=16,
        help='Point grid of the automatic mask generator per (upscaled) patch (default 16: ~32 px spacing '
             'in 512 px patches, enough for ~58 px fibres at 50x; 32 found no more fibres in the sweep '
             'but takes ~3x longer).'
    )

    parser.add_argument(
        '--pred_iou_thresh',
        type=float,
        default=0.88,
        help='AMG: drop masks with predicted IoU below this (default tuned for the fine-tuned checkpoint; '
             'use ~0.1 for the base model, whose IoU head is not calibrated for fibres).'
    )

    parser.add_argument(
        '--stability_score_thresh',
        type=float,
        default=0.8,
        help='AMG: drop masks whose stability score (IoU under logit-threshold shifts) is below this.'
    )

    parser.add_argument(
        '--box_nms_thresh',
        type=float,
        default=0.2,
        help='AMG: box IoU for non-maximum suppression of duplicate masks within a patch '
             '(0.1 suppresses some close neighbours).'
    )

    parser.add_argument(
        '--min_mask_region_area',
        type=int,
        default=150,
        help='AMG: remove holes / islands smaller than this many px from each mask.'
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
        '--min_bright_fraction',
        type=float,
        default=0.0,
        help='Discard objects with less than this fraction of their pixels brighter than '
             '--bright_threshold (fibres are bright, the matrix between them dark; e.g. 0.5). 0 = off.'
    )

    parser.add_argument(
        '--bright_threshold',
        type=float,
        default=None,
        help='Grey value separating bright fibres from the dark matrix. Default: Otsu threshold per '
             'image (inside the mask in mask-aware mode). The used value is printed and saved in the .h5.'
    )

    parser.add_argument(
        '--min_circularity',
        type=float,
        default=0.0,
        help='Discard stitched instances with circularity 4*pi*A/P^2 below this (1 = disk; '
             'e.g. 0.75 keeps nearly round fibre cross-sections). 0 = off.'
    )

    parser.add_argument(
        '--min_axis_ratio',
        type=float,
        default=0.0,
        help='Discard stitched instances whose minor/major axis ratio is below this (1 = round, '
             'e.g. 0.6 drops elongated shapes). 0 = off.'
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
             "Requires a '<stem>.mask.<ext>' file next to each image; errors out if missing."
    )

    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help="Explicit mask file for a single --input image (e.g. data/.../01.mask.tif). "
             "Implies --mask_aware. Without it, '<stem>.mask.<ext>' next to the image is used."
    )

    parser.add_argument(
        '--mask_min_inside',
        type=float,
        default=0.5,
        help='Mask-aware mode: reject detected objects with less than this fraction of their pixels '
             'inside the mask (default 0.5; 1.0 = only objects completely inside, 0 = off).'
    )

    parser.add_argument(
        '--mask_patch_size',
        type=int,
        default=512,
        help='Patch size (square) used for mask-aware tiling.'
    )

    parser.add_argument(
        '--mask_stride',
        type=int,
        default=256,
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

    if args.mask is not None:
        if args.dir_flag:
            parser.error("--mask needs a single image as --input; for a directory the "
                         "'<stem>.mask.<ext>' companions are used with --mask_aware.")
        if not os.path.isfile(args.mask):
            parser.error(f"mask file '{args.mask}' not found.")
        args.mask_aware = True


    # If output not given → create results folder in input path
    if args.output is None:
        if args.dir_flag:
            args.output = os.path.join(args.input, "results")
        else:
            args.output = os.path.join(os.path.dirname(args.input), "results")
        
        os.makedirs(args.output, exist_ok=True)

    main(args)