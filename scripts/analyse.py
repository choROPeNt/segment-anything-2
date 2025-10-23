from __future__ import annotations

import sys

import warnings
warnings.filterwarnings("ignore", message=".*MPS.*fallback.*")

import os,random
import argparse
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # quite nice life hack
import numpy as np

import torch


import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Please use the new API settings to control TF32 behavior.*",
)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import matplotlib.pyplot as plt
from matplotlib import colormaps


from PIL import Image
import h5py
from tqdm import tqdm


from sam2.addons import Sam2Patcher
from sam2.addons import show_anns, write_h5


seed = 3
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def select_device():
    """
    short function to select the device for computation (CUDA/CPU/MPS)
    returns the selected torch device
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("matmul:", torch.backends.cuda.matmul.fp32_precision)
        print("cudnn conv:", torch.backends.cudnn.conv.fp32_precision)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
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

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=48, # correspond to 16**2 = 256 detection points which is similar to fibers per patch
            points_per_batch=96, # Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
            pred_iou_thresh=0.1,
            min_mask_region_area=150,
            box_nms_thresh=0.1,
            stability_score_thresh=0.95,  # Reduce the stability threshold to keep more masks
    )
    
    ## intialize tiler 
    patcher = Sam2Patcher(
        patch_h=1024, 
        patch_w=1024, 
        overlap_h=256, 
        overlap_w=256, 
        pad_mode="constant",
        merge_iou=0.1
    )




    for i, (file,file_name, file_path) in enumerate(zip(files,file_names,file_paths)):
        print("#----------------------------")
        print(f"Processing file {i+1}/{len(files)}: {file}")
        image, was_grayscale  = load_image_from_path(file_path)

        ## Patching
        patches, offsets, padded_shape = patcher.tile_numpy(image)
        print(f"Tiled into {len(patches)} patches of shape {patches[0].shape}, \
              padded shape: {padded_shape}")
        
        # Pre-Allocate some Space
        h5_path = os.path.join(output_path,f"{file_name}.patches.result.h5")

        delete_h5_if_exists(h5_path)

        #----- Mask Generation -----
        all_detections = []
        for p in tqdm(patches,
            total=len(patches),
            desc="Mask generation",
            unit="patch"
        ):
            ## detect masks per patch
            detections = mask_generator.generate(p)
            ## gather results
            if args.max_area:
                max_area = args.max_area
                filterd = [d for d in detections if d["area"] < max_area]
                all_detections.append(filterd)
            else:
                all_detections.append(detections)

        

        label_map, instances = patcher.stitch_sam2_instances(
            padded_shape_hw=padded_shape[:2],
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