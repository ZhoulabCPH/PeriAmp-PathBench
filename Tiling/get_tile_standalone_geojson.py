"""
Label tiles against tumor annotations using the grid from get_tile_standalone_stream.py.
Output filenames and level-0 tile coordinates match the stream script's CSV-only output. Pyramid-level and MPP modes give the same coordinates when their downsample factors match.
Use legacy_coord_offset in pyramid-level mode to add the historical offset (y0 - x0, x0 - y0), or convert old CSVs with migrate_tile_coords.py.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import PIL.ImageDraw
from matplotlib import collections as mpl_collections, patches as mpl_patches, pyplot as plt
from openslide import OpenSlide
from monai.data.wsi_reader import WSIReader
from tqdm import tqdm

# Reuse the streaming tiler's coordinate and foreground logic.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import get_tile_standalone_stream as base  # noqa: E402


# Default settings
input_wsi_dirs = [
    r"data/annotated_slides/Ampullary",
    r"data/annotated_slides/Cholangiocarcinoma",
    r"data/annotated_slides/Duodenal",
    r"data/annotated_slides/Pancreatic"
        # One or more WSI directories
]
geojson_dir: Optional[str] = None  # None looks for annotations next to each slide.
output_dir = r"outputs/interpretability/tumor_area_output"  # Root for CSVs, thumbnails, and patches

patch_level = 0  # Pyramid level used by the tiler
tile_size = 256  # Tile size used by the tiler
use_mpp_mode = False  # Enable MPP-based tiling.
target_mpp: Optional[float] = 0.5  # Target microns per pixel
occupancy_threshold = 0.1  # Minimum foreground occupancy
foreground_threshold: Optional[float] = None  # None selects Otsu thresholding.
roi_margin = 0  # ROI padding in pixels

export_patches = False  # Save annotated tiles as PNG files.
patch_export_dir: Optional[str] = None  # None uses output_dir/tumor_patches.
save_thumbnails = True  # Save a thumbnail with annotation contours and the ROI.
thumbnail_dir: Optional[str] = None  # None uses output_dir/thumbnails.
thumbnail_size = 2048  # Thumbnail long edge in pixels
thumbnail_draw_outside = True  # Also draw foreground tiles outside the annotation.

annotation_mode = "overlap"  # Classify tiles by overlap fraction or center point.
annotation_overlap_threshold = 0.5  # Minimum area fraction for overlap mode
annotation_scale = 1.0  # Scale GeoJSON coordinates when they are outside level-0 pixel units.
annotation_classes: Optional[List[str]] = None  # None includes every annotation class.
annotation_mask_downsample: Optional[int] = None  # None selects the raster downsample automatically.
annotation_max_mask_pixels = 64_000_000  # Maximum number of annotation-mask pixels

legacy_coord_offset = True  # Add the historical offset for compatibility with legacy pyramid-mode CSVs.
require_annotation = True  # Skip slides without GeoJSON annotations.
skip_existing = True  # Skip slides that already have a nonempty CSV.
verbose_csv = False  # Include occupancy, overlap_ratio, read_x, and read_y columns.
tile_progress = True

VALID_SLIDE_SUFFIXES = (".svs", ".tif", ".tiff", ".mdsx", ".ndpi")
ANNOTATION_SUFFIXES = (".geojson", ".json")


# GeoJSON parsing
Ring = np.ndarray  # (N, 2) float array in level-0 pixels
Polygon = Tuple[Ring, List[Ring]]  # (Exterior ring, list of interior rings)


def _ring_from_coords(coords: Sequence[Sequence[float]], scale: float) -> Optional[Ring]:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return None
    arr = arr[:, :2] * float(scale)
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[:1]])
    return arr


def _polygons_from_geometry(geometry: Dict[str, Any], scale: float) -> List[Polygon]:
    if not isinstance(geometry, dict):
        return []
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    polygons: List[Polygon] = []
    if geom_type == "Polygon" and coords:
        exterior = _ring_from_coords(coords[0], scale)
        if exterior is not None:
            holes = [r for r in (_ring_from_coords(c, scale) for c in coords[1:]) if r is not None]
            polygons.append((exterior, holes))
    elif geom_type == "MultiPolygon" and coords:
        for poly_coords in coords:
            polygons.extend(_polygons_from_geometry({"type": "Polygon", "coordinates": poly_coords}, scale))
    elif geom_type == "GeometryCollection":
        for sub in geometry.get("geometries", []):
            polygons.extend(_polygons_from_geometry(sub, scale))
    elif geom_type in ("LineString", "LinearRing") and coords:
        # QuPath can export a closed contour as a LineString. Treat it as a polygon.
        exterior = _ring_from_coords(coords, scale)
        if exterior is not None:
            polygons.append((exterior, []))
    return polygons


def _feature_class_name(feature: Dict[str, Any]) -> Optional[str]:
    props = feature.get("properties") or {}
    classification = props.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    if props.get("name"):
        return str(props["name"])
    return None


def load_annotation_polygons(geojson_path: Union[str, Path], scale: float = 1.0,
                             class_filter: Optional[Sequence[str]] = None) -> List[Polygon]:
    """Read GeoJSON polygons in level-0 pixel coordinates."""
    with open(geojson_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        features = data
    elif data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"geometry": data, "properties": {}}]

    wanted = {c.strip().lower() for c in class_filter} if class_filter else None
    polygons: List[Polygon] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        if wanted is not None:
            name = _feature_class_name(feature)
            if name is None or name.strip().lower() not in wanted:
                continue
        geometry = feature.get("geometry", feature)
        polygons.extend(_polygons_from_geometry(geometry, scale))
    return polygons


class AnnotationRaster:
    """Rasterize polygons into a low-resolution mask to test tile overlap efficiently."""

    def __init__(self, polygons: Sequence[Polygon], slide_wh: Tuple[int, int], downsample: int) -> None:
        self.downsample = int(max(1, downsample))
        slide_w, slide_h = int(slide_wh[0]), int(slide_wh[1])
        self.width = max(1, int(np.ceil(slide_w / self.downsample)))
        self.height = max(1, int(np.ceil(slide_h / self.downsample)))

        fill_img = PIL.Image.new("L", (self.width, self.height), 0)
        hole_img = PIL.Image.new("L", (self.width, self.height), 0)
        fill_draw = PIL.ImageDraw.Draw(fill_img)
        hole_draw = PIL.ImageDraw.Draw(hole_img)
        for exterior, holes in polygons:
            fill_draw.polygon([tuple(p) for p in (exterior / self.downsample)], fill=1)
            for hole in holes:
                hole_draw.polygon([tuple(p) for p in (hole / self.downsample)], fill=1)
        self.mask = np.asarray(fill_img, dtype=bool) & ~np.asarray(hole_img, dtype=bool)
        self.area_px = int(self.mask.sum()) * self.downsample ** 2

    def _tile_slice(self, x0: float, y0: float, side: float) -> Tuple[int, int, int, int]:
        mx0 = int(round(x0 / self.downsample))
        my0 = int(round(y0 / self.downsample))
        span = max(1, int(round(side / self.downsample)))
        return mx0, my0, mx0 + span, my0 + span

    def overlap_ratio(self, x0: float, y0: float, side: float) -> float:
        mx0, my0, mx1, my1 = self._tile_slice(x0, y0, side)
        denominator = float((mx1 - mx0) * (my1 - my0))
        cx0, cy0 = max(0, mx0), max(0, my0)
        cx1, cy1 = min(self.width, mx1), min(self.height, my1)
        if cx0 >= cx1 or cy0 >= cy1:
            return 0.0
        return float(self.mask[cy0:cy1, cx0:cx1].sum()) / denominator

    def contains_center(self, x0: float, y0: float, side: float) -> bool:
        mx = int((x0 + side / 2.0) / self.downsample)
        my = int((y0 + side / 2.0) / self.downsample)
        if 0 <= mx < self.width and 0 <= my < self.height:
            return bool(self.mask[my, mx])
        return False


def choose_mask_downsample(slide_wh: Tuple[int, int], l0_tile_side: float, requested: Optional[int],
                           max_pixels: int) -> int:
    """Choose a mask downsample that limits memory while keeping each tile about 32 pixels wide or larger."""
    if requested is not None and requested > 0:
        downsample = int(requested)
    else:
        downsample = max(1, int(l0_tile_side // 32))
    slide_w, slide_h = slide_wh
    while (slide_w / downsample) * (slide_h / downsample) > max_pixels:
        downsample *= 2
    return downsample


# Tile reads use the streaming tiler's padding rules.
def read_tile_at_level(slide: OpenSlide, rel_xy: Sequence[int], tile_px: int, level: int, coord_downsample: float,
                       origin_xy: Sequence[int], roi_hw: Tuple[int, int]) -> np.ndarray:
    """Return a CHW uint8 tile in pyramid-level mode. Pad beyond the ROI with white and let OpenSlide handle out-of-bounds pixels within the ROI, matching the MONAI reader."""
    rel_x, rel_y = int(rel_xy[0]), int(rel_xy[1])
    roi_h, roi_w = int(roi_hw[0]), int(roi_hw[1])
    canvas = np.full((tile_px, tile_px, 3), 255, dtype=np.uint8)

    cx0, cy0 = max(0, rel_x), max(0, rel_y)
    cx1, cy1 = min(roi_w, rel_x + tile_px), min(roi_h, rel_y + tile_px)
    if cx0 >= cx1 or cy0 >= cy1:
        return np.moveaxis(canvas, -1, 0)

    loc_x = int(round(origin_xy[0] + cx0 * coord_downsample))
    loc_y = int(round(origin_xy[1] + cy0 * coord_downsample))
    region = np.asarray(slide.read_region((loc_x, loc_y), level, (cx1 - cx0, cy1 - cy0)))[:, :, :3]
    canvas[cy0 - rel_y:cy1 - rel_y, cx0 - rel_x:cx1 - rel_x, :] = region
    return np.moveaxis(canvas, -1, 0)


def read_window_level0(slide: OpenSlide, loc_l0_xy: Sequence[int], size_px: int) -> np.ndarray:
    """Read a square level-0 window as CHW uint8 pixels and pad regions outside the slide with white, using the same rules as the streaming MPP reader."""
    x0, y0 = int(loc_l0_xy[0]), int(loc_l0_xy[1])
    slide_w, slide_h = slide.dimensions
    canvas = np.full((size_px, size_px, 3), 255, dtype=np.uint8)
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(slide_w, x0 + size_px), min(slide_h, y0 + size_px)
    if ix0 >= ix1 or iy0 >= iy1:
        return np.moveaxis(canvas, -1, 0)
    region = np.asarray(slide.read_region((ix0, iy0), 0, (ix1 - ix0, iy1 - iy0)))[:, :, :3]
    canvas[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, :] = region
    return np.moveaxis(canvas, -1, 0)


# Thumbnail rendering
def render_annotation_thumbnail(slide_path: Union[str, Path], polygons: Sequence[Polygon],
                                tile_records: Sequence[Dict[str, Any]], l0_tile_side: float,
                                out_path: Union[str, Path], thumb_size: int = 2048,
                                draw_outside: bool = False) -> None:
    """Draw red annotation contours, green tumor-region tiles, and a dashed yellow ROI bounding box on the slide thumbnail."""
    with OpenSlide(str(slide_path)) as slide_obj:
        slide_w, slide_h = slide_obj.dimensions
        scale = float(thumb_size) / max(slide_w, slide_h)
        thumb = slide_obj.get_thumbnail((max(1, int(slide_w * scale)), max(1, int(slide_h * scale))))
    thumb_arr = np.asarray(thumb.convert("RGB"))
    sx = thumb_arr.shape[1] / float(slide_w)
    sy = thumb_arr.shape[0] / float(slide_h)

    fig, ax = plt.subplots(figsize=(thumb_arr.shape[1] / 150.0, thumb_arr.shape[0] / 150.0), dpi=150)
    ax.imshow(thumb_arr)
    ax.set_axis_off()

    inside = [r for r in tile_records if r["in_annotation"]]
    outside = [r for r in tile_records if not r["in_annotation"]]

    def _rects(records):
        return [mpl_patches.Rectangle((r["read_x"] * sx, r["read_y"] * sy), l0_tile_side * sx, l0_tile_side * sy)
                for r in records]

    if draw_outside and outside:
        ax.add_collection(mpl_collections.PatchCollection(
            _rects(outside), facecolor="none", edgecolor="#8c8c8c", linewidths=0.15, alpha=0.6))
    if inside:
        ax.add_collection(mpl_collections.PatchCollection(
            _rects(inside), facecolor="#2ca02c", edgecolor="#145214", linewidths=0.15, alpha=0.35))
        xs = np.array([r["read_x"] for r in inside], dtype=np.float64)
        ys = np.array([r["read_y"] for r in inside], dtype=np.float64)
        bx0, by0 = xs.min() * sx, ys.min() * sy
        bx1, by1 = (xs.max() + l0_tile_side) * sx, (ys.max() + l0_tile_side) * sy
        ax.add_patch(mpl_patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                           edgecolor="#ffb000", linestyle="--", linewidth=1.2))

    for exterior, holes in polygons:
        ax.plot(exterior[:, 0] * sx, exterior[:, 1] * sy, color="red", linewidth=1.2)
        for hole in holes:
            ax.plot(hole[:, 0] * sx, hole[:, 1] * sy, color="darkorange", linewidth=1.0, linestyle="--")

    ax.set_title(f"{Path(slide_path).name}  tumor tiles: {len(inside)} / {len(tile_records)}", fontsize=9)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logging.info(f"Saving annotation thumbnail {out_path}")


# Slide annotation
def get_annotation_csv_path(save_dir: Union[str, Path], slide_id: str) -> Path:
    return Path(save_dir) / f"{slide_id}_patch_annotation.csv"


def find_annotation_file(slide_path: Path, anno_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    search_dir = Path(anno_dir) if anno_dir else slide_path.parent
    for suffix in ANNOTATION_SUFFIXES:
        candidate = search_dir / (slide_path.stem + suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def process_one_slide(slide_path: Path, geojson_path: Path, cfg: Dict[str, Any]) -> Optional[Path]:
    slide_id = slide_path.name  # Keep the file extension to match the tiler.
    out_dir = Path(cfg["output_dir"])
    csv_path = get_annotation_csv_path(out_dir, slide_id)
    if cfg["skip_existing"] and base.is_csv_check_already_processed(csv_path):
        print(f">>> Skipping {csv_path} - already processed")
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the same ROI and foreground threshold as the tiler.
    roi_level = 0 if cfg["use_mpp_mode"] else int(cfg["patch_level"])
    loader = base.LoadROId(WSIReader(backend="OpenSlide"), level=roi_level, margin=int(cfg["roi_margin"]),
                           foreground_threshold=cfg["foreground_threshold"], read_image=False)
    sample = loader({"image": str(slide_path), "slide_id": slide_id, "metadata": {}})
    origin_xy = (int(sample["origin_xy"][0]), int(sample["origin_xy"][1]))
    fg_threshold = float(sample["foreground_threshold"])
    roi_h, roi_w = int(sample["roi_size_hw"][0]), int(sample["roi_size_hw"][1])
    tile_size_out = int(cfg["tile_size"])

    # Use the tiler's coordinate layer and grid rules.
    coord_downsample = float(sample["scale"])
    effective_use_mpp = False
    if cfg["use_mpp_mode"]:
        base_mpp_value = base.read_slide_mpp(slide_path)
        target = cfg["target_mpp"]
        if base_mpp_value is not None and target:
            coord_downsample = float(target) / float(base_mpp_value)
            effective_use_mpp = True
            print(f"MPP mode: base_mpp={base_mpp_value}, target_mpp={target}, mpp_scale={coord_downsample}")
        else:
            print("MPP mode requested but MPP/target_mpp unavailable, falling back to level-based tiling.")

    # The historical offset (y0 - x0, x0 - y0) applies only to pyramid-level tiling.
    delta_xy = (0, 0)
    if cfg["legacy_coord_offset"]:
        if effective_use_mpp:
            raise ValueError('legacy_coord_offset only applies to pyramid-level tiling. The legacy MPP grid itself was shifted, and '
                             'a coordinate offset cannot reproduce its crops. Use the corrected coordinates.')
        delta_xy = (origin_xy[1] - origin_xy[0], origin_xy[0] - origin_xy[1])
        print(f"Legacy coordinate offset enabled: CSV coords = true coords + {delta_xy}")

    # Rasterize annotations. l0_tile_side is the tile side length at level 0.
    l0_tile_side = float(max(1, int(round(tile_size_out * coord_downsample))))
    polygons = load_annotation_polygons(geojson_path, scale=float(cfg["annotation_scale"]),
                                        class_filter=cfg["annotation_classes"])
    if not polygons:
        print(f"!!! No usable polygon found in {geojson_path}, skipping {slide_id}.")
        return None
    with OpenSlide(str(slide_path)) as slide_obj:
        slide_wh = slide_obj.dimensions
    mask_ds = choose_mask_downsample(slide_wh, l0_tile_side, cfg["annotation_mask_downsample"],
                                     int(cfg["annotation_max_mask_pixels"]))
    raster = AnnotationRaster(polygons, slide_wh, mask_ds)
    print(f"Annotation: {len(polygons)} polygon(s), raster {raster.width}x{raster.height} "
          f"(downsample={mask_ds}), area ~{raster.area_px / 1e6:.1f} Mpx")

    # Generate coordinates with the same scaling and grid functions as the tiler.
    if effective_use_mpp:
        grid_h, grid_w = base.scale_roi_size_hw(roi_h, roi_w, coord_downsample)
    else:
        grid_h, grid_w = roi_h, roi_w
    grid_coords, _ = base.compute_grid_coords(grid_h, grid_w, tile_size_out)
    source_px = int(l0_tile_side)

    export_dir: Optional[Path] = None
    dataset_csv_file = None
    if cfg["export_patches"]:
        export_dir = Path(cfg["patch_export_dir"]) / slide_id
        export_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv_file = (export_dir / "dataset.csv").open("w", newline="", encoding="utf-8")
        dataset_csv_file.write("slide_id,tile_id,image,label,tile_x,tile_y,occupancy\n")

    columns = ["filename", "tile_x", "tile_y", "in_annotation"]
    if cfg["verbose_csv"]:
        columns += ["occupancy", "overlap_ratio", "read_x", "read_y"]

    tile_records: List[Dict[str, Any]] = []
    n_total = 0
    n_inside = 0
    use_center = str(cfg["annotation_mode"]).lower() == "center"
    overlap_th = float(cfg["annotation_overlap_threshold"])

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file, OpenSlide(str(slide_path)) as slide_obj:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for rel_loc in tqdm(grid_coords, f"Tiles ({slide_id[:6]}…)", unit="img", disable=not cfg["tile_progress"]):
            loc_l0 = base.rel_to_level0_xy(rel_loc, coord_downsample, origin_xy)
            read_x, read_y = int(loc_l0[0]), int(loc_l0[1])
            if effective_use_mpp:
                tile_chw = read_window_level0(slide_obj, (read_x, read_y), source_px)
            else:
                tile_chw = read_tile_at_level(slide_obj, rel_loc, tile_size_out, roi_level, coord_downsample,
                                              origin_xy, (grid_h, grid_w))
            fg_mask, _ = base.segment_foreground(tile_chw, fg_threshold)
            occupancy = float(fg_mask.mean(dtype=np.float32))
            if occupancy <= float(cfg["occupancy_threshold"]):
                continue

            out_x = read_x + delta_xy[0]
            out_y = read_y + delta_xy[1]

            overlap = raster.overlap_ratio(read_x, read_y, l0_tile_side)
            if use_center:
                in_annotation = int(raster.contains_center(read_x, read_y, l0_tile_side))
            else:
                in_annotation = int(overlap >= overlap_th)
            n_total += 1
            n_inside += in_annotation

            filename = base.get_patch_filename((out_x, out_y))
            row = [filename, out_x, out_y, in_annotation]
            if cfg["verbose_csv"]:
                row += [f"{occupancy:.6f}", f"{overlap:.6f}", read_x, read_y]
            writer.writerow(row)

            if cfg["save_thumbnails"]:
                tile_records.append({"read_x": read_x, "read_y": read_y, "in_annotation": in_annotation})

            if in_annotation and export_dir is not None:
                tile_to_save = tile_chw
                if source_px != tile_size_out:
                    tile_to_save = base.resize_chw_tile(tile_chw, tile_size_out)
                base.save_image(tile_to_save, export_dir / filename)
                tile_id = base.get_tile_id(slide_id, (out_x, out_y))
                dataset_csv_file.write(f"{slide_id},{tile_id},{slide_id}/{filename},,{out_x},{out_y},{occupancy}\n")

    if dataset_csv_file is not None:
        dataset_csv_file.close()

    if cfg["save_thumbnails"]:
        thumb_dir = Path(cfg["thumbnail_dir"])
        thumb_dir.mkdir(parents=True, exist_ok=True)
        try:
            base.save_thumbnail(slide_path, thumb_dir / (slide_id + "_original.png"), size_target=int(cfg["thumbnail_size"]))
            render_annotation_thumbnail(slide_path, polygons, tile_records, l0_tile_side,
                                        thumb_dir / (slide_id + "_annotation_tiles.png"),
                                        thumb_size=int(cfg["thumbnail_size"]),
                                        draw_outside=bool(cfg["thumbnail_draw_outside"]))
        except Exception:
            logging.exception("Failed to render annotation thumbnail")

    ratio = (n_inside / n_total * 100.0) if n_total else 0.0
    print(f"Slide {slide_id}: {n_total} tissue tiles, {n_inside} in annotation ({ratio:.2f}%). CSV -> {csv_path}")
    if export_dir is not None:
        print(f"Exported {n_inside} tumor patches to {export_dir}")
    return csv_path


def build_config(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "input_wsi_dirs": list(input_wsi_dirs),
        "geojson_dir": geojson_dir,
        "output_dir": output_dir,
        "patch_level": patch_level,
        "tile_size": tile_size,
        "use_mpp_mode": use_mpp_mode,
        "target_mpp": target_mpp,
        "occupancy_threshold": occupancy_threshold,
        "foreground_threshold": foreground_threshold,
        "roi_margin": roi_margin,
        "export_patches": export_patches,
        "patch_export_dir": patch_export_dir,
        "save_thumbnails": save_thumbnails,
        "thumbnail_dir": thumbnail_dir,
        "thumbnail_size": thumbnail_size,
        "thumbnail_draw_outside": thumbnail_draw_outside,
        "annotation_mode": annotation_mode,
        "annotation_overlap_threshold": annotation_overlap_threshold,
        "annotation_scale": annotation_scale,
        "annotation_classes": annotation_classes,
        "annotation_mask_downsample": annotation_mask_downsample,
        "annotation_max_mask_pixels": annotation_max_mask_pixels,
        "legacy_coord_offset": legacy_coord_offset,
        "require_annotation": require_annotation,
        "skip_existing": skip_existing,
        "verbose_csv": verbose_csv,
        "tile_progress": tile_progress,
        "slide": None,
    }
    if args is not None:
        for key, value in vars(args).items():
            if value is not None:
                cfg[key] = value
    # Accept a list of input directories or a comma-separated string.
    raw_dirs = cfg.get("input_wsi_dirs")
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]
    cfg["input_wsi_dirs"] = [p.strip() for raw in (raw_dirs or []) for p in str(raw).split(",") if p.strip()]
    cfg["patch_export_dir"] = cfg["patch_export_dir"] or str(Path(cfg["output_dir"]) / "tumor_patches")
    cfg["thumbnail_dir"] = cfg["thumbnail_dir"] or str(Path(cfg["output_dir"]) / "thumbnails")
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Label WSI tiles using GeoJSON tumor annotations.')
    parser.add_argument("--slide", help='Process a single WSI file')
    parser.add_argument("--input-wsi-dirs", "--input-wsi-dir", dest="input_wsi_dirs", nargs="+",
                        help='Input WSI directories, separated by spaces or commas')
    parser.add_argument("--geojson-dir", dest="geojson_dir", help='Annotation directory. Defaults to the slide directory.')
    parser.add_argument("--output-dir", dest="output_dir", help='Output root directory')
    parser.add_argument("--patch-level", dest="patch_level", type=int, help='Pyramid level')
    parser.add_argument("--tile-size", dest="tile_size", type=int, help='Tile size in pixels')
    parser.add_argument("--use-mpp-mode", dest="use_mpp_mode", action="store_true", default=None, help='Enable MPP-based tiling')
    parser.add_argument("--target-mpp", dest="target_mpp", type=float, help='Target microns per pixel')
    parser.add_argument("--occupancy-threshold", dest="occupancy_threshold", type=float, help='Minimum foreground occupancy')
    parser.add_argument("--foreground-threshold", dest="foreground_threshold", type=float, help='Foreground brightness threshold. Defaults to Otsu thresholding.')
    parser.add_argument("--roi-margin", dest="roi_margin", type=int, help='ROI padding in pixels')
    parser.add_argument("--export-patches", dest="export_patches", action="store_true", default=None,
                        help='Export tiles inside the tumor annotation')
    parser.add_argument("--patch-export-dir", dest="patch_export_dir", help='Tile export directory')
    parser.add_argument("--save-thumbnails", dest="save_thumbnails", action="store_true", default=None,
                        help='Save thumbnails with annotation contours and the tumor ROI')
    parser.add_argument("--thumbnail-dir", dest="thumbnail_dir", help='Thumbnail directory')
    parser.add_argument("--thumbnail-size", dest="thumbnail_size", type=int, help='Thumbnail long edge in pixels')
    parser.add_argument("--thumbnail-draw-outside", dest="thumbnail_draw_outside", action="store_true", default=None,
                        help='Draw foreground tiles outside the annotation as well')
    parser.add_argument("--annotation-mode", dest="annotation_mode", choices=["overlap", "center"], help='Tile inclusion rule')
    parser.add_argument("--annotation-overlap-threshold", dest="annotation_overlap_threshold", type=float,
                        help='Minimum overlap fraction in overlap mode')
    parser.add_argument("--annotation-scale", dest="annotation_scale", type=float, help='Annotation coordinate scale factor')
    parser.add_argument("--annotation-classes", dest="annotation_classes", nargs="+", help='Include only the named annotation classes')
    parser.add_argument("--annotation-mask-downsample", dest="annotation_mask_downsample", type=int,
                        help='Annotation mask downsample. Defaults to automatic selection.')
    parser.add_argument("--legacy-coord-offset", dest="legacy_coord_offset", action="store_true", default=None,
                        help='Add the historical coordinate offset to match legacy pyramid-mode CSVs')
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", default=None,
                        help='Reprocess slides with existing CSVs')
    parser.add_argument("--verbose-csv", dest="verbose_csv", action="store_true", default=None,
                        help='Include occupancy, overlap_ratio, and read coordinates in the CSV')
    parser.add_argument("--no-progress", dest="tile_progress", action="store_false", default=None, help='Disable the progress bar')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = build_config(build_arg_parser().parse_args(argv))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    if cfg["slide"]:
        slides = [Path(cfg["slide"])]
    else:
        slides = []
        for wsi_dir in cfg["input_wsi_dirs"]:
            dir_path = Path(wsi_dir)
            if not dir_path.is_dir():
                print(f"!!! Input WSI dir not found: {dir_path} - skipped.")
                continue
            slides.extend(sorted(p for p in dir_path.iterdir()
                                 if p.suffix.lower() in VALID_SLIDE_SUFFIXES))
    # Keep the first slide with each filename to avoid overwriting output CSVs.
    seen_names = set()
    unique_slides = []
    for p in slides:
        if p.name in seen_names:
            print(f"!!! Duplicate slide name {p.name}: {p} ignored (output would collide).")
            continue
        seen_names.add(p.name)
        unique_slides.append(p)
    slides = unique_slides
    if not slides:
        print("No WSI found.")
        return

    for slide_path in slides:
        print(f"Processing: {slide_path.name}")
        anno_path = find_annotation_file(slide_path, cfg["geojson_dir"])
        if anno_path is None:
            message = f"!!! Annotation (.geojson/.json) not found for {slide_path.name}"
            if cfg["require_annotation"]:
                print(message + " - skipped.")
                continue
            print(message)
            continue
        try:
            process_one_slide(slide_path, anno_path, cfg)
        except Exception:
            logging.exception(f"Failed to process {slide_path}")
    print("All WSIs annotated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
