"""
Stream WSI tiles using the GigaPath foreground segmentation, ROI, grid, and occupancy steps.
LoadROId supplies origin in (y, x) order for MONAI and origin_xy in (x, y) order for coordinate arithmetic. Older outputs carry a shift of (y0 - x0, x0 - y0). Use migrate_tile_coords.py to convert eligible artifacts.
Pyramid-level and MPP modes share the same grid. The former uses the pyramid downsample and the latter uses target_mpp / base_mpp. Equal factors yield equal coordinates.
"""

import os
import csv
import shutil
import tempfile
import traceback
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import PIL
import skimage.filters
from openslide import OpenSlide
from monai.data.wsi_reader import WSIReader
from tqdm import tqdm
from matplotlib import collections, patches, pyplot as plt


# Default settings
input_wsi_dir = r"data/slides"  # Input WSI directory
tile_save_dir = r"outputs/tiles/stream"  # Tile output directory
patch_level = 0  # Pyramid level, with 0 at full resolution
tile_size = 256  # Tile size in pixels

# MPP-based tiling
# When use_mpp_mode is enabled:
# Read the physical resolution from openslide.mpp-x or TIFF XResolution.
# Read the ROI at level 0 and calculate the source window size from target_mpp.
# Resample the window to the target MPP even when no exact pyramid level exists.
use_mpp_mode = True  # Enable MPP-based tiling.
target_mpp: Optional[float] = 0.5  # Target microns per pixel. A value of 0.5 is about 20x.

# CSV-only validation mode
# Apply the usual tile filters and omit PNG and thumbnail exports.
# Write one CSV per slide with the same filenames and coordinates as a full tiling run.
csv_check_only = False  # Write only the coordinate CSV when enabled.


# Source: gigapath/preprocessing/data/box_utils.py L15-L146
# Box helpers
@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0:
            raise ValueError(f"Width must be strictly positive, received {self.w}")
        if self.h <= 0:
            raise ValueError(f"Height must be strictly positive, received {self.w}")

    def __add__(self, shift: Sequence[int]) -> 'Box':
        if len(shift) != 2:
            raise ValueError("Shift must be two-dimensional")
        return Box(x=self.x + shift[0], y=self.y + shift[1], w=self.w, h=self.h)

    def __mul__(self, factor: float) -> 'Box':
        return Box(x=int(self.x * factor), y=int(self.y * factor), w=int(self.w * factor), h=int(self.h * factor))

    def __rmul__(self, factor: float) -> 'Box':
        return self * factor

    def __truediv__(self, factor: float) -> 'Box':
        return self * (1.0 / factor)

    def add_margin(self, margin: int) -> 'Box':
        return Box(x=self.x - margin, y=self.y - margin, w=self.w + 2 * margin, h=self.h + 2 * margin)

    def clip(self, other: 'Box') -> Optional['Box']:
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x + self.w, other.x + other.w)
        y1 = min(self.y + self.h, other.y + other.h)
        try:
            return Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
        except ValueError:
            return None

    def to_slices(self) -> Tuple[slice, slice]:
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w))

    @staticmethod
    def from_slices(slices: Sequence[slice]) -> 'Box':
        vert_slice, horz_slice = slices
        return Box(x=horz_slice.start, y=vert_slice.start, w=horz_slice.stop - horz_slice.start, h=vert_slice.stop - vert_slice.start)


# Source: gigapath/preprocessing/data/box_utils.py L129-L146
def get_bounding_box(mask: np.ndarray) -> Box:
    from scipy import ndimage
    if mask.ndim != 2:
        raise TypeError(f"Expected a 2D array but got an array with shape {mask.shape}")
    slices = ndimage.find_objects(mask > 0)
    if not slices:
        raise RuntimeError("The input mask is empty")
    assert len(slices) == 1
    return Box.from_slices(slices[0])


# Array tiling and assembly
# Source: gigapath/preprocessing/data/Tiling.py L15-L19
def get_1d_padding(length: int, tile_size_: int) -> Tuple[int, int]:
    pad = (tile_size_ - length % tile_size_) % tile_size_
    return (pad // 2, pad - pad // 2)


# Source: gigapath/preprocessing/data/Tiling.py L21-L43
def pad_for_tiling_2d(array: np.ndarray, tile_size_: int, channels_first: bool = True, **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    height, width = array.shape[1:] if channels_first else array.shape[:-1]
    padding_h = get_1d_padding(height, tile_size_)
    padding_w = get_1d_padding(width, tile_size_)
    padding = [padding_h, padding_w]
    channels_axis = 0 if channels_first else 2
    padding.insert(channels_axis, (0, 0))
    padded_array = np.pad(array, padding, **pad_kwargs)
    offset = (padding_w[0], padding_h[0])
    return padded_array, np.array(offset)


# Source: gigapath/preprocessing/data/Tiling.py L45-L86
def tile_array_2d(array: np.ndarray, tile_size_: int, channels_first: bool = True, **pad_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(array, tile_size_, channels_first, **pad_kwargs)
    if channels_first:
        channels, height, width = padded_array.shape
    else:
        height, width, channels = padded_array.shape
    n_tiles_h = height // tile_size_
    n_tiles_w = width // tile_size_

    if channels_first:
        intermediate_shape = (channels, n_tiles_h, tile_size_, n_tiles_w, tile_size_)
        axis_order = (1, 3, 0, 2, 4)
        output_shape = (n_tiles_h * n_tiles_w, channels, tile_size_, tile_size_)
    else:
        intermediate_shape = (n_tiles_h, tile_size_, n_tiles_w, tile_size_, channels)
        axis_order = (0, 2, 1, 3, 4)
        output_shape = (n_tiles_h * n_tiles_w, tile_size_, tile_size_, channels)

    tiles = padded_array.reshape(intermediate_shape)
    tiles = tiles.transpose(axis_order)
    tiles = tiles.reshape(output_shape)

    coords_h = tile_size_ * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size_ * np.arange(n_tiles_w) - offset_w
    coords = np.stack(np.meshgrid(coords_w, coords_h), axis=-1).reshape(-1, 2)
    return tiles, coords


# Source: gigapath/preprocessing/data/Tiling.py L89-L130
def assemble_tiles_2d(tiles: np.ndarray, coords: np.ndarray, fill_value: Optional[float] = np.nan, channels_first: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if coords.shape[0] != tiles.shape[0]:
        raise ValueError(f"Tile coordinates and values must have the same length, got {coords.shape[0]} and {tiles.shape[0]}")
    if channels_first:
        n_tiles, channels, tile_size_, _ = tiles.shape
    else:
        n_tiles, tile_size_, _, channels = tiles.shape
    tile_xs, tile_ys = coords.T
    x_min, x_max = min(tile_xs), max(tile_xs + tile_size_)
    y_min, y_max = min(tile_ys), max(tile_ys + tile_size_)
    width = x_max - x_min
    height = y_max - y_min
    output_shape = (channels, height, width) if channels_first else (height, width, channels)
    array = np.full(output_shape, fill_value)
    offset = np.array([-x_min, -y_min])
    for idx in range(n_tiles):
        row = coords[idx, 1] + offset[1]
        col = coords[idx, 0] + offset[0]
        if channels_first:
            array[:, row:row + tile_size_, col:col + tile_size_] = tiles[idx]
        else:
            array[row:row + tile_size_, col:col + tile_size_, :] = tiles[idx]
    return array, offset


# Foreground segmentation and WSI loading
# Source: gigapath/preprocessing/data/foreground_segmentation.py L23-L31
def get_luminance(slide: np.ndarray) -> np.ndarray:
    # Use float32 to avoid overflow when summing large windows or batches.
    return slide.mean(axis=-3, dtype=np.float32)


# Source: gigapath/preprocessing/data/foreground_segmentation.py L33-L47
def segment_foreground(slide: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
    luminance = get_luminance(slide)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
        logging.info(f"Otsu threshold from luminance: {threshold}")
    return luminance < threshold, float(threshold)


# Source: gigapath/preprocessing/data/slide_utils.py L66-L85 (compatibility helper)
def _get_image_size(img, size=None, level=None, location=(0, 0), backend="openslide"):
    max_size = []
    downsampling_factor = []
    if backend == "openslide":
        downsampling_factor = img.level_downsamples[level]
        max_size = img.level_dimensions[level][::-1]
    elif backend == "cucim":
        downsampling_factor = img.resolutions["level_downsamples"][level]
        max_size = img.resolutions["level_dimensions"][level][::-1]
    elif backend == "tifffile":
        level0_size = img.pages[0].shape[:2]
        max_size = img.pages[level].shape[:2]
        downsampling_factor = np.mean([level0_size[i] / max_size[i] for i in range(len(max_size))])
    level_location = [round(location[i] / downsampling_factor) for i in range(len(location))]
    size = [max_size[i] - level_location[i] for i in range(len(max_size))]
    return size


# Source: gigapath/preprocessing/data/slide_utils.py L88-L104
def load_slide_at_level(reader: WSIReader, slide_obj: OpenSlide, level: int) -> np.ndarray:
    size = _get_image_size(slide_obj, level=level)
    img_data, meta_data = reader.get_data(slide_obj, size=size, level=level)
    logging.info(f"img: {img_data.dtype} {img_data.shape}, metadata: {meta_data}")
    return img_data


# Source: gigapath/preprocessing/data/slide_utils.py L113-L181
class LoadROId:
    def __init__(self, image_reader: WSIReader, image_key: str = "image", level: int = 0, margin: int = 0, foreground_threshold: Optional[float] = None, read_image: bool = True) -> None:
        self.image_reader = image_reader
        self.image_key = image_key
        self.level = level
        self.margin = margin
        self.foreground_threshold = foreground_threshold
        self.read_image = read_image

    def _get_bounding_box(self, slide_obj: OpenSlide) -> Tuple[Box, float]:
        highest_level = slide_obj.level_count - 1
        slide = load_slide_at_level(self.image_reader, slide_obj, level=highest_level)
        if slide_obj.level_count == 1:
            logging.warning("Only one image level found. segment_foregound will use a lot of memory.")
        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)
        scale = slide_obj.level_downsamples[highest_level]
        bbox = scale * get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox, threshold

    def __call__(self, data: Dict) -> Dict:
        logging.info(f"LoadROId: read {data[self.image_key]}")
        image_obj: OpenSlide = self.image_reader.read(data[self.image_key])
        logging.info("LoadROId: get bbox")
        level0_bbox, threshold = self._get_bounding_box(image_obj)
        logging.info(f"LoadROId: level0_bbox: {level0_bbox}")
        scale = image_obj.level_downsamples[self.level]
        scaled_bbox = level0_bbox / scale
        # Pass origin in (row, column) order to MONAI get_data(location=...).
        # Use origin_xy for (x, y) arithmetic to avoid shifting coordinates by (y0 - x0, x0 - y0).
        origin = (level0_bbox.y, level0_bbox.x)
        origin_xy = (level0_bbox.x, level0_bbox.y)
        get_data_kwargs = dict(location=origin, size=(scaled_bbox.h, scaled_bbox.w), level=self.level)
        if self.read_image:
            img_data, _ = self.image_reader.get_data(image_obj, **get_data_kwargs)
            logging.info(f"img_data: {img_data.dtype} {img_data.shape}")
            data[self.image_key] = img_data
        else:
            # Populate ROI metadata without loading the slide pixels.
            data["roi_size_hw"] = (scaled_bbox.h, scaled_bbox.w)
        data.update(get_data_kwargs)
        data["origin"] = origin
        data["origin_xy"] = origin_xy
        data["scale"] = scale
        data["foreground_threshold"] = threshold
        image_obj.close()
        return data


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


# Stream tiles to limit memory usage.
def generate_tiles_stream(slide_image: np.ndarray, tile_size_: int, foreground_threshold: float, occupancy_threshold: float, channels_first: bool = True) -> Iterable[Tuple[np.ndarray, np.ndarray, float]]:
    """Yield (tile_chw, rel_location_xy, occupancy) for tiles that pass the foreground threshold without holding the full tile array in memory."""
    padded_array, (offset_w, offset_h) = pad_for_tiling_2d(slide_image, tile_size_, channels_first, constant_values=255)
    if channels_first:
        channels, height, width = padded_array.shape
    else:
        height, width, channels = padded_array.shape
    n_tiles_h = height // tile_size_
    n_tiles_w = width // tile_size_

    coords_h = tile_size_ * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size_ * np.arange(n_tiles_w) - offset_w

    for ih in range(n_tiles_h):
        for iw in range(n_tiles_w):
            rel_x = coords_w[iw]
            rel_y = coords_h[ih]
            row = ih * tile_size_
            col = iw * tile_size_
            if channels_first:
                tile = padded_array[:, row:row + tile_size_, col:col + tile_size_]
            else:
                tile = padded_array[row:row + tile_size_, col:col + tile_size_, :]
                tile = np.moveaxis(tile, -1, 0)

            # Use the same occupancy threshold as the batch path.
            fg_mask, _ = segment_foreground(tile, foreground_threshold)
            occupancy = float(fg_mask.mean(dtype=np.float32))
            if occupancy > occupancy_threshold:
                yield tile, np.array([rel_x, rel_y]), occupancy


# MPP helpers
def read_slide_mpp(slide_path: Union[str, Path]) -> Optional[float]:
    """Read microns per pixel from OpenSlide metadata, then fall back to TIFF XResolution. Return None when neither is usable."""
    try:
        with OpenSlide(str(slide_path)) as os_obj:
            props = os_obj.properties
            mpp_x = props.get('openslide.mpp-x', None)
            if mpp_x is not None:
                return float(mpp_x)
            tiff_xres = props.get('tiff.XResolution', None)
            if tiff_xres is not None:
                # Convert XResolution from pixels per centimeter to microns per pixel.
                # Divide 10,000 microns per centimeter by pixels per centimeter.
                return (1.0 / float(tiff_xres)) * 10000.0
            # Warn when both metadata fields are unavailable.
            warn_msg = f"!!! MPP metadata not found (openslide.mpp-x/tiff.XResolution). Slide: {slide_path}"
            print(warn_msg)
            logging.warning(warn_msg)
    except Exception:
        logging.exception("Failed to read MPP from slide properties")
    return None


def compute_source_tile_size_px(base_mpp: float, target_mpp_value: float, output_tile_size_px: int) -> int:
    """Compute the source window size as output pixels multiplied by target_mpp / base_mpp."""
    scale = target_mpp_value / base_mpp
    src_size = int(round(output_tile_size_px * scale))
    return max(1, src_size)


def resize_chw_tile(tile_chw: np.ndarray, dest_size: int) -> np.ndarray:
    """Resize a CHW tile to a square of the requested size and return it in CHW order."""
    h, w = tile_chw.shape[1], tile_chw.shape[2]
    if h == dest_size and w == dest_size:
        return tile_chw
    arr_hwc = np.moveaxis(tile_chw, 0, -1).astype(np.uint8)
    pil_img = PIL.Image.fromarray(arr_hwc)
    pil_resized = pil_img.resize((dest_size, dest_size), resample=PIL.Image.BILINEAR)
    arr_resized = np.asarray(pil_resized)
    if arr_resized.ndim == 2:
        arr_resized = np.expand_dims(arr_resized, axis=-1)
    return np.moveaxis(arr_resized, -1, 0)


def compute_grid_coords(height: int, width: int, tile_size_: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Apply pad_for_tiling_2d's center-padding rule and return relative [x, y] coordinates and the width and height offsets."""
    dummy = np.zeros((1, height, width), dtype=np.uint8)
    _, (offset_w, offset_h) = pad_for_tiling_2d(dummy, tile_size_, channels_first=True, constant_values=255)
    n_tiles_h = int(np.ceil((height + 2 * offset_h) / tile_size_))
    n_tiles_w = int(np.ceil((width + 2 * offset_w) / tile_size_))
    coords_h = tile_size_ * np.arange(n_tiles_h) - offset_h
    coords_w = tile_size_ * np.arange(n_tiles_w) - offset_w
    coords = np.stack(np.meshgrid(coords_w, coords_h), axis=-1).reshape(-1, 2)
    return coords, (int(offset_w), int(offset_h))


def scale_roi_size_hw(roi_h_l0: int, roi_w_l0: int, coord_downsample: float) -> Tuple[int, int]:
    """Scale the level-0 ROI dimensions into the pyramid layer or virtual MPP layer using the same rounding as LoadROId."""
    scaled = Box(x=0, y=0, w=int(roi_w_l0), h=int(roi_h_l0)) / float(coord_downsample)
    return int(scaled.h), int(scaled.w)


def rel_to_level0_xy(rel_xy: np.ndarray, coord_downsample: float, origin_xy: Sequence[int]) -> np.ndarray:
    """Convert layer-relative (x, y) coordinates into the level-0 top-left position. Both tiling modes use this conversion."""
    rel = np.asarray(rel_xy, dtype=np.float64)
    return (float(coord_downsample) * rel + np.asarray(origin_xy, dtype=np.float64)).astype(int)


def read_region_with_padding(slide: Union[str, Path, OpenSlide], location_xy: Tuple[int, int], size_hw: Tuple[int, int]) -> np.ndarray:
    """Read a level-0 window with white padding and return CHW uint8 pixels. Pass an open OpenSlide handle while streaming to avoid reopening the WSI for every tile."""
    if not isinstance(slide, OpenSlide):
        with OpenSlide(str(slide)) as os_obj:
            return read_region_with_padding(os_obj, location_xy, size_hw)

    x0, y0 = int(location_xy[0]), int(location_xy[1])
    h, w = int(size_hw[0]), int(size_hw[1])
    W, H = slide.dimensions
    x1, y1 = x0 + w, y0 + h
    # Destination canvas
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    # Intersection with the slide bounds
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)
    if ix0 < ix1 and iy0 < iy1:
        # Convert the PIL region to a NumPy array and select its RGB channels.
        sub = np.asarray(slide.read_region((ix0, iy0), 0, (ix1 - ix0, iy1 - iy0)))[:, :, :3]
        # Place the read window on the destination canvas.
        dx0 = ix0 - x0
        dy0 = iy0 - y0
        canvas[dy0:dy0 + (iy1 - iy0), dx0:dx0 + (ix1 - ix0), :] = sub
    return np.moveaxis(canvas, -1, 0)


def choose_vis_level(slide_path: Union[str, Path], roi_size_hw: Tuple[int, int], max_pixels: int = 8_000_000) -> Tuple[int, Tuple[int, int]]:
    """Return the first pyramid level whose ROI fits max_pixels and its height and width. Fall back to the coarsest level."""
    with OpenSlide(str(slide_path)) as os_obj:
        h0, w0 = int(roi_size_hw[0]), int(roi_size_hw[1])
        best_level = os_obj.level_count - 1
        best_size = (max(1, int(round(h0 / os_obj.level_downsamples[best_level]))),
                     max(1, int(round(w0 / os_obj.level_downsamples[best_level]))))
        for level in range(os_obj.level_count - 1, -1, -1):
            ds = float(os_obj.level_downsamples[level])
            h = max(1, int(round(h0 / ds)))
            w = max(1, int(round(w0 / ds)))
            if h * w <= max_pixels:
                return level, (h, w)
        return best_level, best_size


def visualize_roi_lowres(slide_path: Union[str, Path], origin_l0_xy: Tuple[int, int], roi_size_hw: Tuple[int, int], tile_info_list: List[Dict[str, Any]], l0_tile_side: int, out_path: Union[str, Path], max_pixels: int = 8_000_000) -> None:
    """Draw the ROI and tile bounds at low resolution. The origin is level-0 (x, y), the ROI dimensions are (height, width), and each tile side is measured in level-0 pixels."""
    with OpenSlide(str(slide_path)) as os_obj:
        vis_level, (h_vis, w_vis) = choose_vis_level(slide_path, roi_size_hw, max_pixels)
        # OpenSlide.read_region expects (x, y) and (width, height).
        x0, y0 = int(origin_l0_xy[0]), int(origin_l0_xy[1])
        # Convert to NumPy to select RGB channels.
        region = np.asarray(os_obj.read_region((x0, y0), vis_level, (int(w_vis), int(h_vis))))[:, :, :3]
        ds = float(os_obj.level_downsamples[vis_level])

    fig, ax = plt.subplots()
    ax.imshow(region)
    rects = []
    for info in tile_info_list:
        # CSV coordinates are level-0 tile corners.
        tx = float(info["tile_x"]) - x0
        ty = float(info["tile_y"]) - y0
        # Map the level-0 position to the visualization layer.
        xy = (tx / ds, ty / ds)
        rects.append(patches.Rectangle(xy, l0_tile_side / ds, l0_tile_side / ds))
    if rects:
        pc = collections.PatchCollection(rects, match_original=True, alpha=0.4, edgecolor="black")
        pc.set_array(np.array([100] * len(rects)))
        ax.add_collection(pc)
    fig.savefig(out_path)
    plt.close()


# Tile generation and slide processing
# Source: gigapath/preprocessing/data/create_tiles_dataset.py L30-L43
def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if occupancy_threshold < 0.0 or occupancy_threshold > 1.0:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    # Use higher precision to avoid float16 overflow in large windows.
    occupancy = foreground_mask.mean(axis=(-2, -1), dtype=np.float32)
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L45-L48
def get_tile_descriptor(tile_location: Sequence[int]) -> str:
    return f"{tile_location[0]:05d}x_{tile_location[1]:05d}y"


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L50-L53
def get_tile_id(slide_id: str, tile_location: Sequence[int]) -> str:
    return f"{slide_id}.{get_tile_descriptor(tile_location)}"


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L64-L85 (optional filter, disabled by default)
def check_empty_tiles(tiles: np.ndarray, std_th: int = 5, extreme_value_portion_th: float = 0.5) -> np.ndarray:
    b, c, h, w = tiles.shape
    flattned_tiles = tiles.reshape(b, c, h * w)
    std_rgb = flattned_tiles[:, :, :].std(axis=2)
    std_rgb_mean = std_rgb.mean(axis=1)
    low_std_mask = std_rgb_mean < std_th
    extreme_value_count = ((flattned_tiles == 0)).sum(axis=2)
    extreme_value_proportion = extreme_value_count / (h * w)
    extreme_value_mask = extreme_value_proportion.max(axis=1) > extreme_value_portion_th
    return low_std_mask | extreme_value_mask


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L87-L124
def generate_tiles(slide_image: np.ndarray, tile_size_: int, foreground_threshold: float, occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    image_tiles, tile_locations = tile_array_2d(slide_image, tile_size_, constant_values=255)
    logging.info(f"image_tiles.shape: {image_tiles.shape}, dtype: {image_tiles.dtype}")
    logging.info(f"Tiled {slide_image.shape} to {image_tiles.shape}")
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)
    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    logging.info(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")
    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]
    if len(tile_locations) == 0:
        logging.warn("No tiles selected")
    else:
        logging.info(f"After filtering: min y: {tile_locations[:, 0].min()}, max y: {tile_locations[:, 0].max()}, min x: {tile_locations[:, 1].min()}, max x: {tile_locations[:, 1].max()}")
    return image_tiles, tile_locations, occupancies, int(n_discarded)


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L127-L153
def get_tile_info(sample: Dict[str, Any], occupancy: float, tile_location: Sequence[int], rel_slide_dir: Path) -> Dict[str, Any]:
    slide_id = sample["slide_id"]
    descriptor = get_tile_descriptor(tile_location)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"
    tile_info = {
        "slide_id": slide_id,
        "tile_id": get_tile_id(slide_id, tile_location),
        "image": rel_image_path,
        "label": sample.get("label", None),
        "tile_x": tile_location[0],
        "tile_y": tile_location[1],
        "occupancy": occupancy,
        "metadata": {"slide_" + key: value for key, value in sample["metadata"].items()},
    }
    return tile_info


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L155-L169
def format_csv_row(tile_info: Dict[str, Any], keys_to_save: Iterable[str], metadata_keys: Iterable[str]) -> str:
    tile_slide_metadata = tile_info.pop("metadata")
    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    dataset_row = ','.join(fields)
    return dataset_row


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L192-L198
def save_thumbnail(slide_path: Union[str, Path], output_path: Union[str, Path], size_target: int = 1024) -> None:
    with OpenSlide(str(slide_path)) as openslide_obj:
        scale = size_target / max(openslide_obj.dimensions)
        thumbnail = openslide_obj.get_thumbnail([int(m * scale) for m in openslide_obj.dimensions])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        thumbnail.save(output_path)
        logging.info(f"Saving thumbnail {output_path}, shape {thumbnail.size}")


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L200-L219
def visualize_tile_locations(slide_sample: Dict[str, Any], output_path: Union[str, Path], tile_info_list: List[Dict[str, Any]], tile_size_: int, origin_offset: Sequence[int]) -> None:
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]
    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        # Use sample["origin_xy"] so origin_offset matches the tile_x/tile_y axis order.
        xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor, (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size_, tile_size_))
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L221-L235
def is_already_processed(output_tiles_dir: Path) -> bool:
    if not output_tiles_dir.exists():
        return False
    if len(list(output_tiles_dir.glob("*.png"))) == 0:
        return False
    dataset_csv_path = output_tiles_dir / "dataset.csv"
    try:
        df = pd.read_csv(dataset_csv_path)
    except Exception:
        return False
    return len(df) > 0


def get_csv_check_path(save_dir: Union[str, Path], slide_id: str) -> Path:
    return Path(save_dir) / f"{slide_id}_patch_coords.csv"


def is_csv_check_already_processed(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    return len(df) > 0


def get_patch_filename(tile_location: Sequence[int]) -> str:
    return f"{get_tile_descriptor(tile_location)}.png"


# Source: gigapath/preprocessing/data/create_tiles_dataset.py L237-L355
def process_slide(sample: Dict[str, Any], level: int, margin: int, tile_size_: int, foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path, thumbnail_dir: Path, tile_progress: bool = False, use_mpp_mode_: bool = False, target_mpp_value: Optional[float] = None, csv_check_only_: bool = False) -> Path:
    slide_metadata: Dict[str, Any] = sample["metadata"]
    keys_to_save = ("slide_id", "tile_id", "image", "label", "tile_x", "tile_y", "occupancy")
    metadata_keys = tuple("slide_" + key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)
    slide_id: str = sample["slide_id"]
    rel_slide_dir = Path(slide_id)
    output_tiles_dir = output_dir / rel_slide_dir
    csv_check_path = get_csv_check_path(output_dir.parent, slide_id)
    coords_csv_file = None
    coords_writer = None
    dataset_csv_file = None
    failed_tiles_file = None

    if csv_check_only_:
        csv_check_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f">>> CSV check mode: {csv_check_path}")
        if is_csv_check_already_processed(csv_check_path):
            logging.info(f">>> Skipping {csv_check_path} - already processed")
            return csv_check_path
        print(("filename", "tile_x", "tile_y"))
        coords_csv_file = csv_check_path.open('w', newline='', encoding='utf-8-sig')
        coords_writer = csv.writer(coords_csv_file)
        coords_writer.writerow(["filename", "tile_x", "tile_y"])
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        print(csv_columns)
        logging.info(f">>> Slide dir {output_tiles_dir}")
        if is_already_processed(output_tiles_dir):
            logging.info(f">>> Skipping {output_tiles_dir} - already processed")
            return output_tiles_dir
        output_tiles_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_path = output_tiles_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w', newline='')
        dataset_csv_file.write(','.join(csv_columns) + '\n')
        failed_tiles_csv_path = output_tiles_dir / "failed_tiles.csv"
        failed_tiles_file = failed_tiles_csv_path.open('w', newline='')
        failed_tiles_file.write('tile_id' + '\n')

    n_failed_tiles = 0

    slide_image_path = Path(sample["image"])  # original path
    logging.info(f"Loading slide {slide_id} ...\nFile: {slide_image_path}")

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_slide_image_path = Path(tmp_dir.name) / slide_image_path.name
    logging.info(f">>> Copying {slide_image_path} to {tmp_slide_image_path}")
    shutil.copy(slide_image_path, tmp_slide_image_path)
    sample["image"] = tmp_slide_image_path
    logging.info(f">>> Finished copying {slide_image_path} to {tmp_slide_image_path}")

    if not csv_check_only_:
        save_thumbnail(slide_image_path, thumbnail_dir / (slide_image_path.name + "_original.png"))

    # Read the ROI at level 0 in MPP mode or at the supplied level otherwise.
    roi_level = 0 if use_mpp_mode_ else level
    # In MPP mode, compute the ROI bounds and threshold without loading the pixels.
    loader = LoadROId(WSIReader(backend="OpenSlide"), level=roi_level, margin=margin, foreground_threshold=foreground_threshold, read_image=not use_mpp_mode_)
    sample = loader(sample)

    # Skip full-resolution ROI visualization to keep memory usage low.
    if (not csv_check_only_) and (not use_mpp_mode_):
        slide_image = sample["image"]
        plt.figure()
        plt.imshow(slide_image.transpose(1, 2, 0))
        plt.savefig(thumbnail_dir / (slide_image_path.name + "_roi.png"))
        plt.close()
        logging.info(f"Saving thumbnail {thumbnail_dir / (slide_image_path.name + '_roi.png')}, shape {slide_image.shape}")

    # Choose the source tile size.
    effective_tile_size = tile_size_
    effective_use_mpp = False
    base_mpp_value: Optional[float] = None
    if use_mpp_mode_:
        # Read MPP metadata and fall back to the original pyramid level if needed.
        try:
            # Prefer the copied temporary file to match subsequent reads.
            tmp_slide_path = sample["image"]
        except Exception:
            tmp_slide_path = slide_image_path
        # sample["image"] can be an array here. Use the original path for metadata.
        if isinstance(tmp_slide_path, (str, Path)):
            base_mpp_value = read_slide_mpp(tmp_slide_path)
        else:
            base_mpp_value = read_slide_mpp(slide_image_path)

        if base_mpp_value is not None and target_mpp_value is not None:
            effective_tile_size = compute_source_tile_size_px(base_mpp_value, target_mpp_value, tile_size_)
            effective_use_mpp = True
            logging.info(f"MPP mode: base_mpp={base_mpp_value}, target_mpp={target_mpp_value}, output_tile={tile_size_} -> source_tile={effective_tile_size}")
        else:
            logging.warning("MPP mode requested but MPP/target_mpp unavailable, falling back to level-based Tiling.")

    logging.info(f"Tiling slide {slide_id} ...")
    # MPP uses the streaming path, where LoadROId runs with read_image=False.
    # Keep streaming when MPP metadata is unavailable because sample["image"] is still a file path.
    use_stream = bool(use_mpp_mode_)
    # Use the pyramid downsample or target_mpp / base_mpp as the coordinate layer scale.
    # Both modes use the same grid, so equal factors yield equal coordinates.
    coord_downsample = float(sample["scale"])
    if effective_use_mpp and base_mpp_value and target_mpp_value:
        coord_downsample = float(target_mpp_value) / float(base_mpp_value)
    if not use_stream:
        image_tiles, rel_tile_locations, occupancies, _ = generate_tiles(sample["image"], effective_tile_size, sample["foreground_threshold"], occupancy_threshold)
    else:
        image_tiles = None
        rel_tile_locations = None
        occupancies = None

    # The level-0 position is origin_xy + level_scale * tile_rel_layer.
    if not use_stream:
        tile_locations = rel_to_level0_xy(rel_tile_locations, coord_downsample, sample["origin_xy"])

    if not use_stream:
        n_tiles = image_tiles.shape[0]
    else:
        # Estimate the total tile count for logging, using roi_size_hw in streaming mode.
        H, W = sample["roi_size_hw"]
        n_tiles = (int(np.ceil(H / effective_tile_size)) * int(np.ceil(W / effective_tile_size)))
    logging.info(f"{n_tiles} tiles found")

    tile_info_list = []
    logging.info(f"{'Writing tile coordinates' if csv_check_only_ else 'Saving tiles'} for slide {slide_id} ...")
    if not use_stream:
        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                tile_info = get_tile_info(sample, float(occupancies[i]), tile_locations[i], rel_slide_dir)
                tile_info_list.append(tile_info)
                if csv_check_only_ and coords_writer is not None:
                    coords_writer.writerow([get_patch_filename(tile_locations[i]), int(tile_locations[i][0]), int(tile_locations[i][1])])
                else:
                    # The nonstreaming pyramid path already produces tiles at the output size.
                    tile_to_save = image_tiles[i]
                    save_image(tile_to_save, output_dir / tile_info["image"])
                    dataset_row = format_csv_row(dict(tile_info), keys_to_save, metadata_keys)
                    if dataset_csv_file is not None:
                        dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(tile_locations[i])
                if failed_tiles_file is not None:
                    failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile {get_tile_id(slide_id, tile_locations[i])}: {e}")
    else:
        # Build the streaming grid in the coordinate layer, which can be virtual in MPP mode.
        # Use the same scaling, grid, and level-0 conversion as the pyramid path.
        # Equal coordinate scales give the same tile positions.
        roi_h_l0, roi_w_l0 = sample["roi_size_hw"]
        grid_h, grid_w = scale_roi_size_hw(roi_h_l0, roi_w_l0, coord_downsample)
        grid_coords, _ = compute_grid_coords(grid_h, grid_w, tile_size_)
        # Resample only when the level-0 tile side differs from tile_size_.
        source_px = max(1, int(round(tile_size_ * coord_downsample)))
        logging.info(f"Stream grid: coord_downsample={coord_downsample}, grid={grid_h}x{grid_w}, "
                     f"tile={tile_size_} (level0 {source_px}px)")
        for rel_loc in tqdm(grid_coords, f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            try:
                loc_L0 = rel_to_level0_xy(rel_loc, coord_downsample, sample["origin_xy"])

                # Read the source window.
                src_tile = read_region_with_padding(slide_image_path, (int(loc_L0[0]), int(loc_L0[1])), (source_px, source_px))
                # occupancy
                fg_mask, _ = segment_foreground(src_tile, sample["foreground_threshold"])
                occ = float(fg_mask.mean(dtype=np.float32))
                if occ <= occupancy_threshold:
                    continue

                tile_info = get_tile_info(sample, occ, loc_L0, rel_slide_dir)
                tile_info_list.append(tile_info)
                if csv_check_only_ and coords_writer is not None:
                    coords_writer.writerow([get_patch_filename(loc_L0), int(loc_L0[0]), int(loc_L0[1])])
                else:
                    if source_px != tile_size_:
                        tile_to_save = resize_chw_tile(src_tile, tile_size_)
                    else:
                        tile_to_save = src_tile
                    save_image(tile_to_save, output_dir / tile_info["image"])
                    dataset_row = format_csv_row(dict(tile_info), keys_to_save, metadata_keys)
                    if dataset_csv_file is not None:
                        dataset_csv_file.write(dataset_row + '\n')
            except Exception as e:
                n_failed_tiles += 1
                descriptor = get_tile_descriptor(loc_L0)
                if failed_tiles_file is not None:
                    failed_tiles_file.write(descriptor + '\n')
                traceback.print_exc()
                warnings.warn(f"An error occurred while saving tile {get_tile_id(slide_id, loc_L0)}: {e}")

        # Render a low-resolution ROI overlay.
        if not csv_check_only_:
            try:
                l0_tile_side = source_px
                visualize_roi_lowres(
                    slide_image_path,
                    origin_l0_xy=sample["origin_xy"],
                    roi_size_hw=(roi_h_l0, roi_w_l0),
                    tile_info_list=tile_info_list,
                    l0_tile_side=l0_tile_side,
                    out_path=thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"),
                )
            except Exception:
                logging.exception("Failed to render low-res ROI tiles visualization")

    if coords_csv_file is not None:
        coords_csv_file.close()
    if dataset_csv_file is not None:
        dataset_csv_file.close()
    if failed_tiles_file is not None:
        failed_tiles_file.close()

    # This full ROI overlay uses an image array from the pyramid path. Streaming renders its own overlay above.
    if (not csv_check_only_) and (not use_stream):
        visualize_tile_locations(sample, thumbnail_dir / (slide_image_path.name + "_roi_tiles.png"), tile_info_list, effective_tile_size, origin_offset=sample["origin_xy"])

    if n_failed_tiles > 0:
        logging.warning(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")
    logging.info(f"Finished processing slide {slide_id}")
    return csv_check_path if csv_check_only_ else output_tiles_dir


# Compatibility entry point for tile_one_slide
# Source: gigapath/pipeline.py L55-L104
def tile_one_slide(slide_file: str = '', save_dir: str = '', level: int = 0, tile_size_: int = 256, use_mpp_mode_: bool = False, target_mpp_value: Optional[float] = None, csv_check_only_: bool = False) -> None:
    slide_id = os.path.basename(slide_file)
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}
    save_path = Path(save_dir)
    if save_path.exists():
        print(f"Warning: Directory {save_path} already exists. ")
    if csv_check_only_:
        print(f"CSV check mode: skip patch/thumbnail export for {slide_file}. Saving coordinate table to {save_path}.")
    if use_mpp_mode_ and target_mpp_value is not None and target_mpp_value > 0:
        # Read base MPP for source-size logging when available.
        base_mpp_print = read_slide_mpp(slide_file)
        if base_mpp_print is not None:
            source_px_print = compute_source_tile_size_px(base_mpp_print, target_mpp_value, tile_size_)
            print(f"Processing slide {slide_file} in MPP mode (target_mpp={target_mpp_value}, base_mpp={base_mpp_print}, tile_size={tile_size_}, source_px={source_px_print}). ROI read level = 0. Saving to {save_path}.")
        else:
            print(f"Processing slide {slide_file} in MPP mode (target_mpp={target_mpp_value}, tile_size={tile_size_}). ROI read level = 0. Saving to {save_path}.")
    else:
        print(f"Processing slide {slide_file} at level {level} with tile size {tile_size_}. Saving to {save_path}.")
    slide_dir = process_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size_=tile_size_,
        foreground_threshold=None,
        occupancy_threshold=0.1,
        output_dir=save_path / "output",
        thumbnail_dir=save_path / "thumbnails",
        tile_progress=True,
        use_mpp_mode_=use_mpp_mode_,
        target_mpp_value=target_mpp_value,
        csv_check_only_=csv_check_only_,
    )
    if csv_check_only_:
        coords_csv_path = slide_dir if slide_dir.suffix.lower() == ".csv" else get_csv_check_path(save_path, slide_id)
        dataset_df = pd.read_csv(coords_csv_path)
        assert len(dataset_df) > 0
        print(f"Slide {slide_file} CSV check done. {len(dataset_df)} tiles recorded in {coords_csv_path}.")
        return
    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0
    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")


# Batch entry point
# Directory traversal adapted from get_tile_mdsx.py L21-L25.
def main() -> None:
    os.makedirs(tile_save_dir, exist_ok=True)
    valid_suffixes = ['.svs', '.tif', '.tiff', '.mdsx']
    wsi_files = [p for p in Path(input_wsi_dir).iterdir() if p.suffix.lower() in valid_suffixes]
    for wsi_path in wsi_files:
        print(f"Processing: {wsi_path.name}")
        tile_one_slide(
            slide_file=str(wsi_path),
            save_dir=tile_save_dir,
            level=patch_level,
            tile_size_=tile_size,
            use_mpp_mode_=use_mpp_mode,
            target_mpp_value=target_mpp,
            csv_check_only_=csv_check_only,
        )
    print("All WSIs tiled successfully.")


if __name__ == "__main__":
    # Set the log level for direct invocation.
    logging.basicConfig(level=logging.INFO)
    main()


