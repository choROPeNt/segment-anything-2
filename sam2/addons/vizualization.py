import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps
import cv2


np.random.seed(3)


def overlay_from_instances(
    anns,
    canvas_shape,               # (H, W)
    alpha=0.35,
    borders=True,
    border_alpha=0.9,
    border_px=1,
    seed=None                   # set for reproducible pseudo-colors if no 'id'
    ):
    """
    Build an RGBA overlay (float32 in [0,1]) from instance annotations, no plotting.

    anns: list of dicts, each with
        - 'segmentation_cropped': (h,w) bool
        - 'bbox': [x0, y0, w, h]  (pixel coords in original image)
      or:
        - 'segmentation': (H,W) bool
      Optional: 'id' (used for stable color hashing), 'area' (for z-order)

    Returns:
        overlay: (H, W, 4) float32, premultiplied alpha NOT applied (straight alpha)
    """
    H, W = map(int, canvas_shape)
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # transparent by default

    if not anns:
        return overlay

    # sort largest first so later small ones sit "on top"
    sorted_anns = sorted(anns, key=lambda a: a.get("area", 0.0), reverse=True)

    rng = np.random.default_rng(seed)

    def color_from_id(obj_id, fallback_i):
        # Stable color by id; fallback to RNG for entries without id
        if obj_id is None:
            r, g, b = rng.random(3)
        else:
            h = abs(hash(int(obj_id))) % (2**32)
            local = np.random.default_rng(h)
            r, g, b = local.random(3)
        return np.array([r, g, b], dtype=np.float32)

    for i, ann in enumerate(sorted_anns):
        seg_full = ann.get("segmentation", None)
        seg_crop = ann.get("segmentation_cropped", None)
        bbox     = ann.get("bbox", None)

        # choose color
        rgb = color_from_id(ann.get("id", None), i)
        rgba = np.concatenate([rgb, [float(alpha)]], axis=0)

        if seg_crop is not None and bbox is not None:
            x0, y0, w, h = map(int, bbox)
            x1, y1 = x0 + w, y0 + h
            # clip bbox to canvas
            x0 = max(0, min(W, x0));  x1 = max(0, min(W, x1))
            y0 = max(0, min(H, y0));  y1 = max(0, min(H, y1))
            if x1 <= x0 or y1 <= y0:
                continue

            m = np.asarray(seg_crop, dtype=bool)
            sub = overlay[y0:y1, x0:x1]      # view
            mh, mw = m.shape
            sh, sw = sub.shape[:2]
            # trim mask if crop is slightly off
            mh2, mw2 = min(mh, sh), min(mw, sw)
            if mh2 <= 0 or mw2 <= 0:
                continue
            m = m[:mh2, :mw2]
            sub = sub[:mh2, :mw2]

            # paint region (straight alpha)
            sub[m, :3] = rgba[:3]
            sub[m, 3]  = rgba[3]

            if borders:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in cnts:
                    if cnt is None or len(cnt) < 2:
                        continue
                    # draw on a small 2D alpha mask then composite
                    border_mask = np.zeros((mh2, mw2), dtype=np.uint8)
                    cv2.drawContours(border_mask, [cnt], -1, color=255, thickness=border_px)
                    bsel = border_mask.astype(bool)
                    sub[bsel, :3] = rgb
                    sub[bsel, 3]  = max(border_alpha, alpha)

            # write back
            overlay[y0:y0+mh2, x0:x0+mw2] = sub

        elif isinstance(seg_full, np.ndarray) and seg_full.ndim == 2:
            m = seg_full.astype(bool)
            if m.shape != (H, W):
                # if full mask is wrong size, skip (or resize cautiously if desired)
                continue

            overlay[m, :3] = rgba[:3]
            overlay[m, 3]  = rgba[3]

            if borders:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                border_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(border_mask, cnts, -1, color=255, thickness=border_px)
                bsel = border_mask.astype(bool)
                overlay[bsel, :3] = rgb
                overlay[bsel, 3]  = max(border_alpha, alpha)

    return overlay





def show_anns(anns, borders=True, canvas_shape=None,alpha=0.5,cmap_key="plasma"):
    """
    Draws annotations on an RGBA canvas.
    
    Each annotation (ann) can have:
      - {'segmentation_crop': (h,w) bool array, 'bbox': [xmin, ymin, width, height]}
      - or {'segmentation': (H,W) bool array}
    
    Args:
        anns: list of dicts
        borders: draw blue outline if True
        canvas_shape: optional (H, W) tuple for global canvas
    """
    if not anns:
        return

    sorted_anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # ---- determine canvas size ----
    if canvas_shape is not None:
        H, W = map(int, canvas_shape)
    else:
        H = W = 0
        for a in sorted_anns:
            if 'segmentation' in a and isinstance(a['segmentation'], np.ndarray):
                H, W = a['segmentation'].shape
                break
        if H == 0 or W == 0:
            for a in sorted_anns:
                if 'bbox' in a and a['bbox'] is not None:
                    xmin, ymin, w, h = map(int, a['bbox'])
                    H = max(H, ymin + h)
                    W = max(W, xmin + w)

    # RGBA canvas
    img = np.ones((H, W, 4), dtype=float)
    img[..., 3] = 0.0  # transparent background

    cmap = colormaps.get_cmap(cmap_key)
    num_colors = len(sorted_anns)
    color_indices = np.random.rand(num_colors)

    for i, ann in enumerate(sorted_anns):
        color = cmap(color_indices[i])
        color_mask = np.array([color[0], color[1], color[2], alpha], dtype=float)  # RGBA with alpha

        seg_crop = ann.get('segmentation_crop', None)
        bbox = ann.get('bbox', None)
        seg_full = ann.get('segmentation', None)

        if seg_crop is not None and bbox is not None:
            # bbox in [xmin, ymin, width, height]
            xmin, ymin, w, h = map(int, bbox)
            xmax = xmin + w
            ymax = ymin + h
            m = np.asarray(seg_crop, dtype=bool)

            # clip to canvas
            xmin = max(0, min(W, xmin))
            ymin = max(0, min(H, ymin))
            xmax = max(0, min(W, xmax))
            ymax = max(0, min(H, ymax))
            if xmax <= xmin or ymax <= ymin:
                continue

            subimg = img[ymin:ymax, xmin:xmax]
            mh, mw = m.shape
            sh, sw = subimg.shape[:2]

            # crop mask if it exceeds canvas region
            if (mh, mw) != (sh, sw):
                m = m[:sh, :sw]

            subimg[m] = color_mask

            if borders:
                # find contours in the mask
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in cnts:
                    if cnt.ndim != 3 or cnt.shape[0] < 2:
                        continue

                    # shift contour coordinates by (xmin, ymin)
                    cnt_shifted = cnt + np.array([[xmin, ymin]])

                    # draw directly on image (in-place)
                    cv2.drawContours(
                        img,                     # image to draw on
                        [cnt_shifted],           # list of contours
                        -1,                      # draw all contours
                        color=(0.5, 0.5, 0.5, alpha),   # mid-gray, full alpha
                        thickness=3,             # 1-pixel wide line
                        lineType=cv2.LINE_AA     # smooth edges
                    )

        elif isinstance(seg_full, np.ndarray) and seg_full.ndim == 2:
            m = seg_full.astype(bool)
            img[m] = color_mask

            if borders:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in cnts:
                    if cnt.ndim != 3 or cnt.shape[0] < 2:
                        continue
                    pts = cnt[:, 0, :]
                    # ax.plot(pts[:, 0], pts[:, 1], linewidth=1.0, color=(0, 0, 1, 0.4))

    ax.imshow(img)
    return (img)