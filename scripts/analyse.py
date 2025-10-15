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
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import matplotlib.pyplot as plt
from matplotlib import colormaps


from PIL import Image
import h5py
from tqdm import tqdm


from sam2.addons import Sam2Patcher
from sam2.addons import show_anns


seed = 3
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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



# def save_patch_group(
#     f: h5py.File,
#     idx: int,
#     patch: np.ndarray,              # (H,W,3) RGB uint8 (what SAM2 used)
#     offset: tuple,                  # (y0, x0)
#     detections: list,               # list of dicts with 'segmentation', 'area', 'bbox', 'predicted_iou'
#     was_grayscale: bool,            # original image was grayscale?
#     compression: str = "lzf"        # "lzf" fast; use "gzip" for smaller files
# ):
#     H, W = patch.shape[:2]
#     g = f.require_group(f"/patches/patch_{idx:03d}")

#     # ---- 1) image (store gray if original was gray) + offset ----
#     if was_grayscale:
#         # BT.709 luma to keep perceptual brightness (uint8)
#         patch_gray = np.clip(
#             0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2],
#             0, 255
#         ).round().astype(np.uint8)

#         # (Re)create dataset as (H,W)
#         need_recreate = "data" in g and g["data"].shape != (H, W)
#         if "data" not in g or need_recreate:
#             if "data" in g: del g["data"]
#             g.create_dataset(
#                 "data", data=patch_gray, compression=compression, shuffle=True,
#                 chunks=(min(256, H), min(256, W))
#             )
#         else:
#             g["data"][...] = patch_gray

#         C_saved = 1

#     else:
#         # Store RGB as-is (H,W,3)
#         need_recreate = "data" in g and g["data"].shape != (H, W, 3)
#         if "data" not in g or need_recreate:
#             if "data" in g: del g["data"]
#             g.create_dataset(
#                 "data", data=patch.astype(np.uint8, copy=False),
#                 compression=compression, shuffle=True,
#                 chunks=(min(256, H), min(256, W), 3)
#             )
#         else:
#             g["data"][...] = patch.astype(np.uint8, copy=False)

#         C_saved = 3

#     if "offset" not in g:
#         g.create_dataset("offset", data=np.asarray(offset, dtype=np.int32))

#     # ---- 2) detections → stacked arrays (N,H,W) + metadata ----
#     segs, areas, bboxes, scores = [], [], [], []
#     for det in (detections or []):
#         m = det.get("segmentation")
#         if m is None:
#             continue
#         m = np.asarray(m, dtype=bool)
#         if m.shape != (H, W):
#             raise ValueError(f"Segmentation shape {m.shape} != patch {(H, W)}")
#         if not m.any():
#             continue

#         segs.append(m.astype(np.uint8))  # 0/1; compresses very well
#         areas.append(int(det.get("area", int(m.sum()))))

#         bbox = det.get("bbox")
#         if bbox is None:
#             ys, xs = np.where(m)
#             x0, y0 = int(xs.min()), int(ys.min())
#             x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
#             bbox = [x0, y0, x1 - x0, y1 - y0]
#         bboxes.append([int(b) for b in bbox])

#         scores.append(float(det.get("predicted_iou", 0.0)))

#     N = len(segs)

#     # (Re)create per-detection datasets to exact sizes
#     for name in ("segmentation", "area", "bbox", "predicted_iou"):
#         if name in g:
#             del g[name]

#     g.create_dataset(
#         "segmentation", shape=(N, H, W), dtype=np.uint8,
#         compression=compression, shuffle=True, chunks=(1, H, W)
#     )
#     if N:
#         g["segmentation"][...] = np.stack(segs, axis=0)

#     g.create_dataset(
#         "area",
#         data=(np.asarray(areas, dtype=np.int32) if N else np.empty((0,), np.int32)),
#         compression=compression, shuffle=True, chunks=(max(1, min(8192, N)),)
#     )
#     g.create_dataset(
#         "bbox",
#         data=(np.asarray(bboxes, dtype=np.int32) if N else np.empty((0, 4), np.int32)),
#         compression=compression, shuffle=True, chunks=(max(1, min(2048, N)), 4)
#     )
#     g.create_dataset(
#         "predicted_iou",
#         data=(np.asarray(scores, dtype=np.float32) if N else np.empty((0,), np.float32)),
#         compression=compression, shuffle=True, chunks=(max(1, min(8192, N)),)
#     )

#     # ---- 3) metadata ----
#     g.attrs["H"] = H
#     g.attrs["W"] = W
#     g.attrs["C_saved"] = C_saved            # 1 for gray, 3 for RGB (on disk)
#     g.attrs["was_grayscale"] = bool(was_grayscale)  # original image info
#     g.attrs["n_detections"] = N

# def load_patch_group(f: h5py.File, idx: int):
#     g = f[f"/patches/patch_{idx:03d}"]
#     img   = g["data"][...]                          # (H,W,C) uint8
#     off   = tuple(g["offset"][...].tolist())        # (y0, x0)
#     seg   = g["segmentation"][...]                  # (N,H,W) uint8
#     area  = g["area"][...].astype(int)
#     bbox  = g["bbox"][...].astype(int)              # (N,4)
#     piou  = g["predicted_iou"][...].astype(float)   # (N,)
#     return img, off, seg.astype(bool), area, bbox, piou

def delete_h5_if_exists(h5_path: str) -> None:
    if os.path.exists(h5_path):
        try:
            os.remove(h5_path)
            print(f"Deleted existing HDF5: {h5_path}")
        except PermissionError as e:
            raise RuntimeError(f"Cannot delete {h5_path} (file may be open elsewhere): {e}")
    else:
        print(f"No existing HDF5 at {h5_path} — nothing to delete.")

def select_device():
    """
    short function to select the device for computation (CUDA/CPU/MPS)
    returns the selected torch device
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
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


def write_h5(path: str, dict_out: dict, overwrite: bool = True):
    """
    Write image/label/mask/instance data to an HDF5 file.

    Expected structure:
    {
        "image": np.ndarray [H, W] or [H, W, C],
        "labels": np.ndarray [H, W],
        "mask": np.ndarray [H, W],
        "instances": [
            {
                "id": int,
                "area": float,
                "bbox": [x0, y0, w, h],
                "segmentation_cropped": np.ndarray (bool)
            },
            ...
        ]
    }

    Parameters
    ----------
    path : str
        Output HDF5 file path.
    dict_out : dict
        Data dictionary as above.
    overwrite : bool
        If True, overwrites existing file.
    """
    if overwrite and os.path.exists(path):
        os.remove(path)

    with h5py.File(path, "w") as h5f:
        # --- scalar / array datasets ---
        for key in ["image", "labels", "mask"]:
            if key not in dict_out:
                continue
            data = np.asarray(dict_out[key])
            h5f.create_dataset(
                key,
                data=data,
                compression="gzip",
                compression_opts=4
            )

        # --- instances ---
        if "instances" in dict_out and dict_out["instances"]:
            grp_inst = h5f.create_group("instances")
            for inst in dict_out["instances"]:
                inst_id = str(inst.get("id", len(grp_inst) + 1)).zfill(4)
                g = grp_inst.create_group(inst_id)

                # Store metadata
                g.create_dataset("id", data=np.int32(inst.get("id", -1)))
                g.create_dataset("area", data=np.float32(inst.get("area", 0.0)))

                bbox = np.asarray(inst.get("bbox", [0, 0, 0, 0]), dtype=np.int32)
                g.create_dataset("bbox", data=bbox)

                seg = np.asarray(inst.get("segmentation_crop", []), dtype=bool)
                g.create_dataset(
                    "segmentation_crop",
                    data=seg,
                    compression="gzip",
                    compression_opts=4
                )

        # Summary
        n_inst = len(dict_out.get("instances", []))
        print(f"✅ Saved: {path}")
        print(f"  ├─ image shape   : {dict_out['image'].shape if 'image' in dict_out else None}")
        print(f"  ├─ labels shape  : {dict_out['labels'].shape if 'labels' in dict_out else None}")
        print(f"  ├─ mask shape    : {dict_out['mask'].shape if 'mask' in dict_out else None}")
        print(f"  └─ instances     : {n_inst}")


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

    sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=24, # correspond to 16**2 = 256 detection points which is similar to fibers per patch
            points_per_batch=64, # Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
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
            print(p.shape)
            ## detect masks per patch
            detections = mask_generator.generate(p)
            ## gather results
            all_detections.append(detections)

        

        label_map, instances = patcher.stitch_sam2_instances(
            padded_shape_hw=padded_shape[:2],
            offsets=offsets,
            per_tile_sam2=all_detections,
            debug=False,
        )

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
        # fig, ax = plt.subplots(2,2, figsize=(20,10))
        # ax = ax.flatten()

        # ax[0].imshow(image,cmap="gray" if was_grayscale else None)  
        # ax[0].axis('off')
        # ax[0].set_title("Input image")

        # ax[1].imshow(label_map,cmap="plasma")
        # ax[1].axis('off')
        # ax[1].set_title(f"Label map - {len(instances)} masks")

        # ax[2].imshow(image,cmap="gray" if was_grayscale else None)
        # instance_map = show_anns(instances, alpha=0.5)
        # ax[2].imshow(instance_map)
        # ax[2].axis('off')
        # ax[2].set_title(f"Overlay - {len(instances)} masks")

        # ax[3].imshow(label_map > 0,cmap="gray")
        # ax[3].axis('off')
        # ax[3].set_title(f"Mask coverage - {100* (label_map > 0).sum()/label_map.size:.1f}% area")


        # plt.tight_layout()
        # plt.show()
        # plt.savefig(os.path.join(output_path, f"{file_name}_results.png"), bbox_inches='tight', dpi=150)


        # plt.close()



            
        #     h5f.create_dataset("image", image, dtype = np.uint8, compression="gzip", shuffle=True)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input data')
    parser.add_argument('--output', type=str, required=False, help='Path to save analysis results')
    args = parser.parse_args()
    
    # Check that input is a directory
    if os.path.isdir(args.input):
        args.dir_flag = True
        print("Input is a directory")
    else:
        args.dir_flag = False
        print("INput is a single file ")


    # If output not given → create results folder in input path
    if args.output is None:
        if args.dir_flag:
            args.output = os.path.join(args.input, "results")
        else:
            args.output = os.path.join(os.path.dirname(args.input), "results")
        
        os.makedirs(args.output, exist_ok=True)

    main(args)