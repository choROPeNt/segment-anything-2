from __future__ import annotations

import gc
import os
import random
import sys
import argparse
import warnings
from typing import cast, Tuple

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colormaps
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
def to_cpu(obj):
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
        points_per_side=96,
        points_per_batch=128,          # lower than 96 to reduce peak memory
        pred_iou_thresh=0.1,
        min_mask_region_area=150,
        box_nms_thresh=0.1,
        stability_score_thresh=0.9,
        # multimask_output=False,       # important
        # output_mode="uncompressed_rle"  # important
    )

def find_cuda_tensors(obj, path="root", found=None):
    if found is None:
        found = []
    import torch

    if torch.is_tensor(obj):
        if obj.is_cuda:
            found.append((path, tuple(obj.shape), obj.dtype, obj.device))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_cuda_tensors(v, f"{path}[{k!r}]", found)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_cuda_tensors(v, f"{path}[{i}]", found)
    return found



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

            # move everything off GPU just in case
            detections = to_cpu(detections)

            if args.max_area:
                filtered = [d for d in detections if d["area"] < args.max_area]
                all_detections.append(filtered)
            else:
                all_detections.append(detections)

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
        '--max_area',
        type=float,
        required=False,
        default=None,
        help='Optional area threshold for filtering small or large regions. Use None to disable filtering.'
    )

    parser.add_argument(
        '--save_fig',
        action='store_true',
        help='If set, saves generated figures instead of only displaying them.'
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