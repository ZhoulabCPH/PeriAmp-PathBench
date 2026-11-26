"""
Convert tile coordinates produced before the ROI origin order was corrected.
Old coordinates included the offset (y0 - x0, x0 - y0) from mixing (y, x) and (x, y). Subtract this per-slide offset to convert pyramid-level coordinates and filenames without changing the tile pixels.
Legacy MPP-mode crops were sampled at shifted locations and must be regenerated to match the new grid. Use --allow-mpp-source only to annotate their coordinates. The offset depends on the slide and foreground threshold and is computed once per WSI.
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openslide import OpenSlide
from monai.data.wsi_reader import WSIReader

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import get_tile_standalone_stream as base  # noqa: E402


# Default settings. Command-line arguments take precedence.
input_wsi_dir = r"data/slides"  # Original WSI directory
output_dir = r"outputs/tiles/coordinate_migration"  # Coordinate map output directory
patch_level = 0  # Legacy pyramid level for grid verification
tile_size = 256  # Legacy tile size for grid verification
roi_margin = 0  # Legacy ROI padding in pixels
foreground_threshold: Optional[float] = None  # Legacy foreground threshold. None selects Otsu thresholding.

VALID_SLIDE_SUFFIXES = (".svs", ".tif", ".tiff", ".mdsx", ".ndpi")
COORD_IN_NAME = re.compile(r"(-?\d+)x_(-?\d+)y")


# Historical offset and corrected grid
def compute_origin_xy(slide_path: Path, margin: int, fg_threshold: Optional[float]) -> Tuple[base.Box, float]:
    """Recompute the level-0 foreground bounding box using the same implementation as LoadROId."""
    loader = base.LoadROId(WSIReader(backend="OpenSlide"), level=0, margin=int(margin),
                           foreground_threshold=fg_threshold, read_image=False)
    with OpenSlide(str(slide_path)) as slide_obj:
        bbox, threshold = loader._get_bounding_box(slide_obj)
    return bbox, float(threshold)


def compute_delta(bbox: base.Box) -> Tuple[int, int]:
    """Return the historical offset (y0 - x0, x0 - y0)."""
    return int(bbox.y - bbox.x), int(bbox.x - bbox.y)


def build_grid_lattice(slide_path: Path, bbox: base.Box, level: int, tile_size_: int) -> set:
    """Rebuild the corrected grid to validate migrated coordinates."""
    with OpenSlide(str(slide_path)) as slide_obj:
        coord_downsample = float(slide_obj.level_downsamples[int(level)])
    grid_h, grid_w = base.scale_roi_size_hw(bbox.h, bbox.w, coord_downsample)
    rel_coords, _ = base.compute_grid_coords(grid_h, grid_w, int(tile_size_))
    origin_xy = (bbox.x, bbox.y)
    lattice = base.rel_to_level0_xy(rel_coords, coord_downsample, origin_xy)
    return {(int(x), int(y)) for x, y in lattice}


# Legacy input readers
def _slide_id_from_csv(csv_path: Path) -> Optional[str]:
    name = csv_path.name
    for suffix in ("_patch_coords.csv", "_patch_annotation.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name == "dataset.csv":
        return csv_path.parent.name
    return None


def load_records_from_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    if "tile_x" not in fieldnames or "tile_y" not in fieldnames:
        raise ValueError(f"{csv_path} lacks tile_x or tile_y")
    return rows, fieldnames


def load_records_from_patch_dir(patch_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for png in sorted(patch_dir.glob("*.png")):
        match = COORD_IN_NAME.search(png.stem)
        if match is None:
            continue
        rows.append({"filename": png.name, "tile_x": match.group(1), "tile_y": match.group(2)})
    if not rows:
        raise ValueError(f"{patch_dir} contains no PNG files with parseable tile coordinates")
    return rows


# Coordinate conversion and outputs
def migrate_one_source(slide_path: Path, source_label: str, rows: List[Dict[str, str]],
                       fieldnames: Optional[List[str]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    slide_id = slide_path.name
    bbox, threshold = compute_origin_xy(slide_path, cfg["roi_margin"], cfg["foreground_threshold"])
    delta_x, delta_y = compute_delta(bbox)
    print(f"{slide_id}: bbox x0={bbox.x} y0={bbox.y} (Otsu={threshold:.2f}) -> Δ=({delta_x}, {delta_y})")

    lattice = None
    if cfg["verify_grid"]:
        lattice = build_grid_lattice(slide_path, bbox, cfg["patch_level"], cfg["tile_size"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / f"{slide_id}_coord_migration.csv"

    n_off_lattice = 0
    n_read_mismatch = 0
    mapping: List[Tuple[str, int, int, str, int, int]] = []
    for row in rows:
        old_x, old_y = int(float(row["tile_x"])), int(float(row["tile_y"]))
        new_x, new_y = old_x - delta_x, old_y - delta_y
        old_name = row.get("filename") or base.get_patch_filename((old_x, old_y))
        new_name = base.get_patch_filename((new_x, new_y))
        if lattice is not None and (new_x, new_y) not in lattice:
            n_off_lattice += 1
        # Use read_x and read_y from the annotation CSV as an independent check.
        if row.get("read_x") not in (None, "") and row.get("read_y") not in (None, ""):
            if (int(float(row["read_x"])), int(float(row["read_y"]))) != (new_x, new_y):
                n_read_mismatch += 1
        mapping.append((old_name, old_x, old_y, new_name, new_x, new_y))

    with mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "old_filename", "old_tile_x", "old_tile_y",
                         "new_filename", "new_tile_x", "new_tile_y", "delta_x", "delta_y"])
        for old_name, old_x, old_y, new_name, new_x, new_y in mapping:
            writer.writerow([slide_id, old_name, old_x, old_y, new_name, new_x, new_y, delta_x, delta_y])
    print(f"  mapping -> {mapping_path} ({len(mapping)} patches)")

    unsafe = bool(n_off_lattice) and not cfg["allow_mpp_source"]
    if n_off_lattice:
        print(f"  !!! {n_off_lattice}/{len(mapping)} converted coordinates fall outside the corrected grid. "
              f"Check that --patch-level, --tile-size, and --roi-margin match the source. "
              f"Regenerate tiles produced by the legacy MPP mode to match the corrected grid.")
        if unsafe:
            print('  Skipped CSV rewriting and tile renaming. Use --allow-mpp-source to request a mapping for legacy MPP output.')
    elif lattice is not None:
        print(f"  Grid verification passed for {len(mapping)} coordinates")
    if n_read_mismatch:
        print(f"  !!! {n_read_mismatch} rows differ from the read_x/read_y columns")

    fixed_path = None
    if cfg["rewrite_csv"] and fieldnames is not None and not unsafe:
        fixed_path = out_dir / f"{Path(source_label).stem}_fixed.csv"
        write_fixed_csv(fixed_path, rows, fieldnames, slide_id, delta_x, delta_y)
        print(f"  fixed csv -> {fixed_path}")

    return {
        "slide_id": slide_id,
        "source": source_label,
        "n_patches": len(mapping),
        "bbox_x0": bbox.x,
        "bbox_y0": bbox.y,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "off_lattice": n_off_lattice,
        "unsafe": int(unsafe),
        "mapping_csv": str(mapping_path),
        "fixed_csv": str(fixed_path) if fixed_path else "",
    }


def write_fixed_csv(fixed_path: Path, rows: List[Dict[str, str]], fieldnames: List[str],
                    slide_id: str, delta_x: int, delta_y: int) -> None:
    """Write corrected coordinates and update derived filename, image, and tile_id columns."""
    with fixed_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            new_row = dict(row)
            old_x, old_y = int(float(row["tile_x"])), int(float(row["tile_y"]))
            new_x, new_y = old_x - delta_x, old_y - delta_y
            new_row["tile_x"] = str(new_x)
            new_row["tile_y"] = str(new_y)
            new_name = base.get_patch_filename((new_x, new_y))
            if "filename" in new_row:
                new_row["filename"] = new_name
            if "image" in new_row and new_row["image"]:
                new_row["image"] = str(Path(new_row["image"]).parent / new_name).replace("\\", "/")
            if "tile_id" in new_row:
                new_row["tile_id"] = base.get_tile_id(slide_id, (new_x, new_y))
            writer.writerow(new_row)


def rename_patch_files(patch_dir: Path, mapping_csv: Path) -> None:
    """Rename patches in two passes using temporary .migrating suffixes so overlapping old and new names do not collide."""
    pairs: List[Tuple[Path, Path]] = []
    with mapping_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src = patch_dir / row["old_filename"]
            dst = patch_dir / row["new_filename"]
            if src.exists():
                pairs.append((src, dst))
    for src, dst in pairs:
        src.rename(dst.with_suffix(dst.suffix + ".migrating"))
    for _, dst in pairs:
        staged = dst.with_suffix(dst.suffix + ".migrating")
        if staged.exists():
            staged.rename(dst)
    print(f"  renamed {len(pairs)} patch files in {patch_dir}")


# Command-line entry point
def find_slide(slide_id: str, wsi_dir: Path) -> Optional[Path]:
    candidate = wsi_dir / slide_id
    if candidate.exists():
        return candidate
    stem = Path(slide_id).stem
    for suffix in VALID_SLIDE_SUFFIXES:
        candidate = wsi_dir / (stem + suffix)
        if candidate.exists():
            return candidate
    return None


def collect_sources(cfg: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """Return (slide_id, source_path) pairs for CSV files or patch directories."""
    sources: List[Tuple[str, Path]] = []
    for raw in cfg["old_csv"] or []:
        path = Path(raw)
        csv_files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
        for csv_file in csv_files:
            if csv_file.name.endswith("_coord_migration.csv") or csv_file.name == "coord_migration_summary.csv":
                continue
            slide_id = cfg["slide_id"] or _slide_id_from_csv(csv_file)
            if slide_id is None:
                print(f"!!! Could not infer the slide ID from {csv_file}. Skipping. Use --slide-id to supply it.")
                continue
            sources.append((slide_id, csv_file))
    for raw in cfg["old_patch_dir"] or []:
        path = Path(raw)
        subdirs = [d for d in sorted(path.iterdir()) if d.is_dir()] if path.is_dir() else []
        patch_dirs = [d for d in subdirs if any(d.glob("*.png"))] or ([path] if any(path.glob("*.png")) else [])
        for patch_dir in patch_dirs:
            sources.append((cfg["slide_id"] or patch_dir.name, patch_dir))
    return sources


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "input_wsi_dir": input_wsi_dir,
        "output_dir": output_dir,
        "patch_level": patch_level,
        "tile_size": tile_size,
        "roi_margin": roi_margin,
        "foreground_threshold": foreground_threshold,
        "old_csv": None,
        "old_patch_dir": None,
        "slide_id": None,
        "rewrite_csv": False,
        "rename_patches": False,
        "verify_grid": True,
        "allow_mpp_source": False,
    }
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert legacy tile coordinates to the corrected grid.')
    parser.add_argument("--input-wsi-dir", dest="input_wsi_dir", help='Original WSI directory for recomputing foreground bounds')
    parser.add_argument("--output-dir", dest="output_dir", help='Output directory for coordinate mappings')
    parser.add_argument("--old-csv", dest="old_csv", nargs="+",
                        help='Legacy CSV file or directory containing coordinate, annotation, or dataset CSV files')
    parser.add_argument("--old-patch-dir", dest="old_patch_dir", nargs="+",
                        help='Legacy PNG directory or a parent containing several slide directories')
    parser.add_argument("--slide-id", dest="slide_id", help='Slide ID to use when it cannot be inferred from the filename')
    parser.add_argument("--patch-level", dest="patch_level", type=int, help='Legacy pyramid level for grid verification')
    parser.add_argument("--tile-size", dest="tile_size", type=int, help='Legacy tile size for grid verification')
    parser.add_argument("--roi-margin", dest="roi_margin", type=int, help='Legacy ROI padding in pixels')
    parser.add_argument("--foreground-threshold", dest="foreground_threshold", type=float,
                        help='Legacy foreground threshold. Defaults to Otsu thresholding.')
    parser.add_argument("--rewrite-csv", dest="rewrite_csv", action="store_true", default=None,
                        help='Write a corrected copy of the source CSV')
    parser.add_argument("--rename-patches", dest="rename_patches", action="store_true", default=None,
                        help='Rename the PNG files in --old-patch-dir in place')
    parser.add_argument("--no-verify-grid", dest="verify_grid", action="store_false", default=None,
                        help='Skip validation against the corrected grid')
    parser.add_argument("--allow-mpp-source", dest="allow_mpp_source", action="store_true", default=None,
                        help='Allow mapping of legacy MPP output even when its sampling grid differs')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = build_config(build_arg_parser().parse_args(argv))
    if not cfg["old_csv"] and not cfg["old_patch_dir"]:
        print('Provide --old-csv, --old-patch-dir, or both.')
        return
    wsi_dir = Path(cfg["input_wsi_dir"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for slide_id, source_path in collect_sources(cfg):
        slide_path = find_slide(slide_id, wsi_dir)
        if slide_path is None:
            print(f"!!! Could not find the WSI for {slide_id} ({wsi_dir}). Skipping.")
            continue
        try:
            fieldnames: Optional[List[str]]
            if source_path.is_dir():
                rows, fieldnames = load_records_from_patch_dir(source_path), None
            else:
                rows, fieldnames = load_records_from_csv(source_path)
            summary = migrate_one_source(slide_path, str(source_path), rows, fieldnames, cfg)
            if cfg["rename_patches"] and source_path.is_dir() and not summary["unsafe"]:
                rename_patch_files(source_path, Path(summary["mapping_csv"]))
            summaries.append(summary)
        except Exception:
            logging.exception(f" Could not process {source_path}")

    if summaries:
        summary_path = Path(cfg["output_dir"]) / "coord_migration_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
