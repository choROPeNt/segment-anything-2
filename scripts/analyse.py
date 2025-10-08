from __future__ import annotations


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

import cv2
from PIL import Image

import json
import h5py
from tqdm import tqdm

seed = 3
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)




from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence


@dataclass
class Sam2Tiler:
    """
    Tiling & stitching helper for SAM2-style instance dicts.

    Parameters
    ----------
    patch_h, patch_w : int
        Patch (tile) height/width in pixels.
    overlap_h, overlap_w : int
        Overlap (stride = patch - overlap). Must satisfy 0 <= overlap < patch.
    pad_mode : {"reflect", "constant"}
        How to pad on the bottom/right if the image is not a multiple of the patch size.
    pad_value : int or float
        Value used for 'constant' padding.
    merge_iou : float
        IoU threshold to merge touching/overlapping instances across tiles.
    """
    patch_h: int
    patch_w: int
    overlap_h: int = 0
    overlap_w: int = 0
    pad_mode: str = "reflect"
    pad_value: float = 0.0
    merge_iou: float = 0.5

    # ----------------------------- public API ---------------------------------

    def tile_numpy(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int, int]]:
        """
        Split an (H, W, 3) image into overlapping patches (N, patch_h, patch_w, 3).

        Returns
        -------
        patches : np.ndarray
            Array of patches with shape (N, patch_h, patch_w, 3).
        offsets : list[tuple[int,int]]
            Top-left (y0, x0) offsets of each patch in *padded* coordinates.
        padded_shape : tuple[int,int,int]
            Shape of the (possibly) padded image used for tiling.
        """
        assert img.ndim == 3 and img.shape[-1] == 3, "Expected (H, W, 3) image"
        H, W, _ = img.shape
        ys = self._positions(H, self.patch_h, self.overlap_h)
        xs = self._positions(W, self.patch_w, self.overlap_w)

        pad_bottom = max(0, (ys[-1] + self.patch_h) - H)
        pad_right  = max(0, (xs[-1] + self.patch_w) - W)
        if pad_bottom or pad_right:
            pad_cfg = ((0, pad_bottom), (0, pad_right), (0, 0))
            if self.pad_mode == "reflect":
                img_p = np.pad(img, pad_cfg, mode="reflect")
            elif self.pad_mode == "constant":
                img_p = np.pad(img, pad_cfg, mode="constant", constant_values=self.pad_value)
            else:
                raise ValueError("pad_mode must be 'reflect' or 'constant'")
        else:
            img_p = img

        patches, offsets = [], []
        for y0 in ys:
            for x0 in xs:
                patches.append(img_p[y0:y0 + self.patch_h, x0:x0 + self.patch_w, :])
                offsets.append((y0, x0))

        return np.stack(patches, axis=0), offsets, img_p.shape

    def stitch_sam2_instances(
                self,
                padded_shape_hw: Tuple[int, int],
                offsets: Sequence[Tuple[int, int]],
                per_tile_sam2: Sequence[List[Dict[str, Any]]],
                debug: bool = False,
                ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Merge per-patch SAM2 detections into global instance & confidence maps.
        Optimized to reduce allocations and per-instance mask work.
        """
        H, W = padded_shape_hw
        inst_map = np.zeros((H, W), dtype=np.int32)
        conf_map = np.zeros((H, W), dtype=np.float32)
        next_id = 1
        total_written = 0

        # tiny epsilon to stabilize score comparisons across backends
        eps = 1e-7

        for (tile_y0, tile_x0), detections in zip(offsets, per_tile_sam2):
            if not detections:
                continue

            for d in detections:
                m = np.asarray(d["segmentation"], dtype=bool)  # no copy if already bool
                if not m.any():
                    continue

                score = float(d.get("predicted_iou", 1.0))
                # bbox in patch coords
                bx, by, bw, bh = d.get("bbox", [0, 0, m.shape[1], m.shape[0]])

                # fallback if bbox missing/degenerate
                if bw <= 0 or bh <= 0:
                    ys, xs = np.where(m)
                    if ys.size == 0:
                        continue
                    by = int(ys.min()); bx = int(xs.min())
                    bh = int(ys.max() - by + 1); bw = int(xs.max() - bx + 1)

                by0 = int(by); bx0 = int(bx)
                by1 = by0 + int(bh); bx1 = bx0 + int(bw)

                # global coords
                gy0 = tile_y0 + by0; gx0 = tile_x0 + bx0
                gy1 = tile_y0 + by1; gx1 = tile_x0 + bx1

                # clamp to canvas
                gy0 = 0 if gy0 < 0 else (H if gy0 > H else gy0)
                gy1 = 0 if gy1 < 0 else (H if gy1 > H else gy1)
                gx0 = 0 if gx0 < 0 else (W if gx0 > W else gx0)
                gx1 = 0 if gx1 < 0 else (W if gx1 > W else gx1)
                if gy0 >= gy1 or gx0 >= gx1:
                    continue

                # restrict mask to bbox once (view)
                m_sub = m[by0:by1, bx0:bx1]
                if not m_sub.any():
                    continue

                # views into global maps for the same region
                im_sub = inst_map[gy0:gy1, gx0:gx1]
                cf_sub = conf_map[gy0:gy1, gx0:gx1]

                # ------ Vectorized overlap + IoU computation ------
                # intersections per existing oid within this bbox region:
                #   count of pixels where m_sub==True and im_sub==oid
                inter_counts = np.bincount(im_sub[m_sub].ravel(), minlength=im_sub.max() + 1)
                # all oids present at intersection (exclude background 0)
                overlapping_oids = np.nonzero(inter_counts[1:])[0] + 1
                best_iou = 0.0
                tgt = None

                if overlapping_oids.size:
                    # area of mask within bbox (constant across oids)
                    area_m = int(m_sub.sum())
                    # pixels of each oid anywhere in bbox (not only where m_sub==True)
                    # this is O(#pixels in bbox), done once
                    region_counts = np.bincount(im_sub.ravel(), minlength=im_sub.max() + 1)

                    # compute IoU(oid) = inter / (area_m + region_area(oid) - inter)
                    inter = inter_counts[overlapping_oids]
                    reg_area = region_counts[overlapping_oids]
                    denom = (area_m + reg_area - inter).astype(np.float32)
                    # avoid divide-by-zero (shouldn’t happen, but safe)
                    denom[denom == 0] = 1.0
                    ious = inter.astype(np.float32) / denom

                    # select best
                    j = int(ious.argmax())
                    best_iou = float(ious[j])
                    tgt = int(overlapping_oids[j])

                # new id if no good match
                if tgt is None or best_iou < self.merge_iou:
                    tgt = next_id
                    next_id += 1

                # Write rule: fill empty OR replace where score >= existing (+eps)
                # Avoids allocating an intermediate big mask twice.
                write = m_sub & ((cf_sub == 0.0) | (score >= cf_sub - eps))
                if write.any():
                    im_sub[write] = tgt
                    cf_sub[write] = score
                    total_written += int(write.sum())

                # views already updated; no need to assign back

        if debug:
            print(f"[stitch] written_px={total_written}, instances={int(inst_map.max())}")

        instances = self._instances_from_maps(inst_map, conf_map)
        return inst_map, conf_map, instances

    # --------------------------- private helpers ------------------------------

    @staticmethod
    def _positions(full: int, win: int, overlap: int) -> List[int]:
        if not (0 <= overlap < win):
            raise ValueError("0 <= overlap < win required")
        if full <= win:
            return [0]
        stride = win - overlap
        pos = [0]
        x = 0
        while True:
            x += stride
            if x + win >= full:
                last = max(0, full - win)
                if last != pos[-1]:
                    pos.append(last)
                break
            pos.append(x)
        return pos

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union)

    @staticmethod
    def _bbox2d(mask_bool: np.ndarray) -> Tuple[List[int], List[int]]:
        ys, xs = np.where(mask_bool)
        if ys.size == 0:
            return [0, 0, 0, 0], [0, 0, 0, 0]
        y0, x0 = int(ys.min()), int(xs.min())
        y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
        w, h = x1 - x0, y1 - y0
        return [x0, y0, w, h], [x0, y0, x1, y1]

    def _instances_from_maps(self, inst_map: np.ndarray, conf_map: np.ndarray) -> List[Dict[str, Any]]:
        """Build SAM2-style dicts for each global instance id (>0)."""
        instances: List[Dict[str, Any]] = []
        ids = np.unique(inst_map)
        ids = ids[ids > 0]
        for iid in ids:
            seg = (inst_map == iid)
            area = int(seg.sum())
            if area == 0:
                continue
            bbox, crop_box = self._bbox2d(seg)
            scores = conf_map[seg]
            predicted_iou = float(scores.mean()) if scores.size else 0.0
            instances.append({
                "segmentation": seg,          # (H,W) bool, global coords
                "area": area,
                "bbox": bbox,                 # [x, y, w, h], global coords
                "predicted_iou": predicted_iou,
                "point_coords": [],           # optional: fill if prompts used
                "stability_score": None,      # optional: fill if computed
                "crop_box": crop_box,         # [x0, y0, x1, y1], global coords
            })
        return instances


def show_anns(anns, borders=True):
    """
    function to display segmentation masks with random colors from the plasma colormap
    anns: list of annotations dict, each containing a 'segmentation' mask and 'area'
    """
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img_shape = sorted_anns[0]['segmentation'].shape
    img = np.ones((*img_shape, 4))  # RGBA image
    img[:, :, 3] = 0  # Fully transparent initially

    cmap = colormaps.get_cmap("plasma")  # Get the plasma colormap
    num_colors = len(sorted_anns)  # Number of different colors needed
    color_indices = np.random.rand(num_colors)  # Randomly sample colors

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color = cmap(color_indices[i])  # Get a random color from plasma colormap
        color_mask = np.array([color[0], color[1], color[2], 0.5])  # RGBA with alpha

        img[m] = color_mask  # Assign color to segmentation mask

        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (1, 1, 1, 0.4), thickness=1)  # Draw white borders

    ax.imshow(img)

def save_patch_group(
    f: h5py.File,
    idx: int,
    patch: np.ndarray,              # (H,W,3) RGB uint8 (what SAM2 used)
    offset: tuple,                  # (y0, x0)
    detections: list,               # list of dicts with 'segmentation', 'area', 'bbox', 'predicted_iou'
    was_grayscale: bool,            # original image was grayscale?
    compression: str = "lzf"        # "lzf" fast; use "gzip" for smaller files
):
    H, W = patch.shape[:2]
    g = f.require_group(f"/patches/patch_{idx:03d}")

    # ---- 1) image (store gray if original was gray) + offset ----
    if was_grayscale:
        # BT.709 luma to keep perceptual brightness (uint8)
        patch_gray = np.clip(
            0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2],
            0, 255
        ).round().astype(np.uint8)

        # (Re)create dataset as (H,W)
        need_recreate = "data" in g and g["data"].shape != (H, W)
        if "data" not in g or need_recreate:
            if "data" in g: del g["data"]
            g.create_dataset(
                "data", data=patch_gray, compression=compression, shuffle=True,
                chunks=(min(256, H), min(256, W))
            )
        else:
            g["data"][...] = patch_gray

        C_saved = 1

    else:
        # Store RGB as-is (H,W,3)
        need_recreate = "data" in g and g["data"].shape != (H, W, 3)
        if "data" not in g or need_recreate:
            if "data" in g: del g["data"]
            g.create_dataset(
                "data", data=patch.astype(np.uint8, copy=False),
                compression=compression, shuffle=True,
                chunks=(min(256, H), min(256, W), 3)
            )
        else:
            g["data"][...] = patch.astype(np.uint8, copy=False)

        C_saved = 3

    if "offset" not in g:
        g.create_dataset("offset", data=np.asarray(offset, dtype=np.int32))

    # ---- 2) detections → stacked arrays (N,H,W) + metadata ----
    segs, areas, bboxes, scores = [], [], [], []
    for det in (detections or []):
        m = det.get("segmentation")
        if m is None:
            continue
        m = np.asarray(m, dtype=bool)
        if m.shape != (H, W):
            raise ValueError(f"Segmentation shape {m.shape} != patch {(H, W)}")
        if not m.any():
            continue

        segs.append(m.astype(np.uint8))  # 0/1; compresses very well
        areas.append(int(det.get("area", int(m.sum()))))

        bbox = det.get("bbox")
        if bbox is None:
            ys, xs = np.where(m)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
            bbox = [x0, y0, x1 - x0, y1 - y0]
        bboxes.append([int(b) for b in bbox])

        scores.append(float(det.get("predicted_iou", 0.0)))

    N = len(segs)

    # (Re)create per-detection datasets to exact sizes
    for name in ("segmentation", "area", "bbox", "predicted_iou"):
        if name in g:
            del g[name]

    g.create_dataset(
        "segmentation", shape=(N, H, W), dtype=np.uint8,
        compression=compression, shuffle=True, chunks=(1, H, W)
    )
    if N:
        g["segmentation"][...] = np.stack(segs, axis=0)

    g.create_dataset(
        "area",
        data=(np.asarray(areas, dtype=np.int32) if N else np.empty((0,), np.int32)),
        compression=compression, shuffle=True, chunks=(max(1, min(8192, N)),)
    )
    g.create_dataset(
        "bbox",
        data=(np.asarray(bboxes, dtype=np.int32) if N else np.empty((0, 4), np.int32)),
        compression=compression, shuffle=True, chunks=(max(1, min(2048, N)), 4)
    )
    g.create_dataset(
        "predicted_iou",
        data=(np.asarray(scores, dtype=np.float32) if N else np.empty((0,), np.float32)),
        compression=compression, shuffle=True, chunks=(max(1, min(8192, N)),)
    )

    # ---- 3) metadata ----
    g.attrs["H"] = H
    g.attrs["W"] = W
    g.attrs["C_saved"] = C_saved            # 1 for gray, 3 for RGB (on disk)
    g.attrs["was_grayscale"] = bool(was_grayscale)  # original image info
    g.attrs["n_detections"] = N

def load_patch_group(f: h5py.File, idx: int):
    g = f[f"/patches/patch_{idx:03d}"]
    img   = g["data"][...]                          # (H,W,C) uint8
    off   = tuple(g["offset"][...].tolist())        # (y0, x0)
    seg   = g["segmentation"][...]                  # (N,H,W) uint8
    area  = g["area"][...].astype(int)
    bbox  = g["bbox"][...].astype(int)              # (N,4)
    piou  = g["predicted_iou"][...].astype(float)   # (N,)
    return img, off, seg.astype(bool), area, bbox, piou

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


def main(args,
        valid_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ):
    
    # Parse arguments
    input_path = args.input
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)
    print(f"Analyzing data in {input_path}, results will be saved to {output_path}")

    # get list of valid files in input path
    files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
    print(f"Found {len(files)} valid files in {input_path}")

    #------------------------
    # Set-Up Model SAM2
    #------------------------
    device = select_device()

    sam2_checkpoint = "/home/dchristi/projects/alpha-capella/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

    mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32, # correspond to 16**2 = 256 detection points which is similar to fibers per patch
            points_per_batch=96, # Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory
            pred_iou_thresh=0.1,
            min_mask_region_area=150,
            box_nms_thresh=0.1,
            stability_score_thresh=0.95,  # Reduce the stability threshold to keep more masks
                )
    
    ## intialize tiler 
    tiler = Sam2Tiler(patch_h=1024, patch_w=1024, overlap_h=256, overlap_w=256, merge_iou=0.5)




    for i, file in enumerate(files):
        print("#----------------------------")
        print(f"Processing file {i+1}/{len(files)}: {file}")
        file_path = os.path.join(input_path, file)
        file_name = os.path.splitext(file)[0]
        
        # Load image
        image = Image.open(file_path)
        # Flag for grayscale
        was_grayscale = False

        if image.mode in ("L", "I"):       # 8-bit or 32-bit grayscale
            was_grayscale = True
            image = image.convert("RGB")   # expand to 3-channel RGB

        elif image.mode == "RGBA":         # keep only RGB, drop alpha
            image = image.convert("RGB")

        # Convert to NumPy array
        image = np.array(image)

        print(f"Shape: {image.shape}, dtype: {image.dtype}")
        print(f"Was grayscale: {was_grayscale}")

        # image = image[450:1474,5000:9048,:] # remove after prototyping
        # print(f"cropped shape: {image.shape},dtype: {image.dtype}")

        #------------------------
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # plt.axis('off')
        # plt.title("Input image")
        # plt.savefig(f"{os.path.splitext(file)[0]}_input.png", bbox_inches='tight', dpi=150)
        # plt.close()
        #------------------------

        ## Patching
        patches, offsets, padded_shape = tiler.tile_numpy(image)
        print(f"Tiled into {len(patches)} patches of shape {patches[0].shape}, padded shape: {padded_shape}")
        
        #------------------------
        # plt.figure(figsize=(10,10))
        # plt.imshow(np.stack(patches,axis=1).reshape((tiler.patch_h, -1, 3)))
        # plt.axis('off')
        # plt.title("Tiled patches")
        # plt.savefig(f"{os.path.splitext(file)[0]}_patches.png", bbox_inches='tight', dpi=150)
        # plt.close()
        #------------------------
        # Pre-Allocate some Space
        h5_path = os.path.join(output_path,f"{file_name}.patches.result.h5")

        delete_h5_if_exists(h5_path)




        ## Run SAM2 on patches
        # all_detections = []
         
        with h5py.File(h5_path, "a") as f:
            
            ## Save original image/panorama to hdf5
            if was_grayscale:
                f.create_dataset("image",data=image[:,:,0],dtype=np.uint8, compression="gzip")
            else:
                f.create_dataset("image",data=image,dtype=np.uint8, compression="gzip")

            ## loop over patches and save them to hdf5
            for j, (p, off) in enumerate(tqdm(zip(patches, offsets),
                                     total=len(patches),
                                     desc="Mask generation",
                                     unit="patch")):

                detections = mask_generator.generate(p)
                             
                # gather results
                # all_detections.append(detections)

                # write to h5-file
                save_patch_group(f, j, p, off, detections, was_grayscale, compression="gzip")

                #------------------------
                # plt.figure(figsize=(10,10))
                # plt.imshow(p)
                # show_anns(detections)
                # plt.axis('off')
                # plt.title(f"Patch {j+1} - {len(detections)} masks")
                # plt.savefig(f"{os.path.splitext(file)[0]}_patch{j+1:02d}_masks.png", bbox_inches='tight', dpi=150)
                # plt.close()
                #------------------------
                





        # print(f"stiching {len(patches)} patches")
        # ## Stitching
        # inst_map, conf_map, instances = tiler.stitch_sam2_instances(
        #     padded_shape_hw=padded_shape[:2],
        #     offsets=offsets,
        #     per_tile_sam2=all_detections,
        #     debug=False,
        # )

        # print(f"inst_map: {inst_map.shape},{inst_map.dtype},{inst_map.nbytes/1024/1024}")
        # print(f"conf_map: {conf_map.shape},{conf_map.dtype},{conf_map.nbytes/1024/1024}")
        # print(f"instances: {len(instances)}, {instances[0]}")

        # print(f"Stitched into {len(instances)} global instances")


        #------------------------
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # show_anns(instances)
        # plt.axis('off')
        # plt.title(f"Stitched - {len(instances)} masks")
        # plt.savefig(f"{os.path.splitext(file)[0]}_stitched_masks.png", bbox_inches='tight', dpi=150)
        # plt.close()
        #------------------------



        # if i == 0:
        #     break # for prototyping






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input data')
    parser.add_argument('--output', type=str, required=False, help='Path to save analysis results')
    args = parser.parse_args()
    
    # Check that input is a directory
    if not os.path.isdir(args.input):
        print(f"Error: --input path '{args.input}' is not a directory.")
        sys.exit(1)

    # If output not given → create results folder in input path
    if args.output is None:
        args.output = args.input
        os.makedirs(args.output, exist_ok=True)

    main(args)