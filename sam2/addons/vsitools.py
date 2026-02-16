from pathlib import Path
from typing import Union, Dict, Any, Tuple
from aicsimageio import AICSImage
from aicsimageio.readers import BioformatsReader
import numpy as np


def read_vsi(
    file_path: Union[str, Path],
    reconstruct_mosaic: bool = False,
    scene: str | None = None,
    load_image: bool = True,
    compute_statistics: bool = False,
    return_numpy: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """
    Load VSI image and extract metadata.

    Parameters
    ----------
    file_path : str | Path
    reconstruct_mosaic : bool
    scene : str | None
    load_image : bool
        If False, only metadata is extracted.
    compute_statistics : bool
        Compute basic stats on first slice.
    return_numpy : bool
        If True, return NumPy array (loads into RAM).
        If False, return Dask array (lazy).

    Returns
    -------
    meta_dict : dict
    image_data : ndarray or dask array or None
    """

    file_path = Path(file_path)

    img = AICSImage(
        file_path,
        reader=BioformatsReader,
        reconstruct_mosaic=reconstruct_mosaic
    )

    if scene is not None:
        img.set_scene(scene)

    meta_dict: Dict[str, Any] = {}

    # --------------------------------------------------
    # Basic info
    # --------------------------------------------------
    meta_dict["file_path"] = str(file_path)
    meta_dict["dims_order"] = str(img.dims)
    meta_dict["shape"] = tuple(img.shape)
    meta_dict["dtype"] = str(img.dtype)

    # --------------------------------------------------
    # Pixel size
    # --------------------------------------------------
    pps = img.physical_pixel_sizes
    meta_dict["physical_pixel_sizes"] = {
        "X": pps.X if pps else None,
        "Y": pps.Y if pps else None,
        "Z": pps.Z if pps else None,
        "unit": "µm (usually)"
    }

    # --------------------------------------------------
    # Scenes / channels
    # --------------------------------------------------
    meta_dict["scenes"] = list(img.scenes)
    meta_dict["current_scene"] = img.current_scene
    meta_dict["channels"] = (
        list(img.channel_names) if img.channel_names else None
    )

    # --------------------------------------------------
    # OME metadata
    # --------------------------------------------------
    ome = img.ome_metadata
    if ome and len(ome.images) > 0:
        pixels = ome.images[0].pixels
        meta_dict["ome"] = {
            "size_x": pixels.size_x,
            "size_y": pixels.size_y,
            "size_z": pixels.size_z,
            "size_c": pixels.size_c,
            "size_t": pixels.size_t,
            "pixel_type": str(pixels.type),
        }

    # --------------------------------------------------
    # Raw metadata
    # --------------------------------------------------
    try:
        meta_dict["bioformats_raw_metadata"] = img.reader.metadata
    except Exception as e:
        meta_dict["bioformats_raw_metadata"] = str(e)

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    image_data = None

    if load_image:
        dask_data = img.get_image_dask_data("TCZYX")

        if return_numpy:
            image_data = dask_data.compute()
        else:
            image_data = dask_data

        if compute_statistics:
            first_slice = image_data[0, 0, 0]
            first_slice_np = np.asarray(first_slice)  # <- makes Pylance happy

            meta_dict["image_statistics"] = {
                "min": float(first_slice_np.min()),
                "max": float(first_slice_np.max()),
                "mean": float(first_slice_np.mean()),
                "std": float(first_slice_np.std()),
            }

    return meta_dict, image_data