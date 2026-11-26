#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose tumor cells that do not match a tile flagged as inside the tumor annotation.
Rebuild the full tiling grid and account for the historical coordinate offset. Classify a missing assignment as an annotation boundary, tissue boundary, grid or ROI boundary, or an unexpected mismatch. Export per-slide diagnostic and summary CSVs and an overlay image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Point,
    Polygon,
    box,
    shape,
)
from shapely.ops import unary_union

from openslide import OpenSlide
from monai.data.wsi_reader import WSIReader


# =====================================================================
# Repository paths
# =====================================================================

_THIS_DIR = Path(__file__).resolve().parent

# Resolve Tiling next to this analysis directory.
# The tiling modules provide the grid and coordinate helpers.
_ROOT = _THIS_DIR.parent
_TILING_DIR = _ROOT / "Tiling"

# Adjust the tiling path for a different checkout layout.
# _TILING_DIR = Path(r"Tiling")

if str(_TILING_DIR) not in sys.path:
    sys.path.insert(0, str(_TILING_DIR))

import Tiling.get_tile_standalone_stream as base  # noqa: E402


# =====================================================================
# Defaults shared with the tiling and cell-assignment workflows
# =====================================================================

DEFAULT_WSI_DIRS = [
    r"data/annotated_slides/Ampullary",
    r"data/annotated_slides/Cholangiocarcinoma",
    r"data/annotated_slides/Duodenal",
    r"data/annotated_slides/Pancreatic",
]

# Cell-to-patch assignment output directory
DEFAULT_ASSIGNMENT_DIR = (
    r"outputs/interpretability/cell_patch_assignment"
)

# Patch annotation CSVs from the GeoJSON tiler
DEFAULT_PATCH_ANNOTATION_DIR = (
    r"outputs/interpretability/tumor_area_output"
)

DEFAULT_OUTPUT_DIR = (
    r"outputs/interpretability/unmatched_cell_diagnostics"
)

# Use cell records to avoid duplicating detection records.
DEFAULT_CELL_SOURCE = "cells"

# Match the settings used to generate the patches.
DEFAULT_PATCH_LEVEL = 0
DEFAULT_TILE_SIZE = 256
DEFAULT_USE_MPP_MODE = False
DEFAULT_TARGET_MPP = 0.5

DEFAULT_OCCUPANCY_THRESHOLD = 0.1
DEFAULT_FOREGROUND_THRESHOLD = None
DEFAULT_ROI_MARGIN = 0

DEFAULT_ANNOTATION_OVERLAP_THRESHOLD = 0.5

# Visualization
DEFAULT_MAX_THUMB = 4000

# Flag cells near the tumor boundary.
# Measure centroid distance to the tumor polygon boundary.
# Set near_tumor_boundary when the distance is at most one patch side.
DEFAULT_BOUNDARY_MARGIN_PX = 256

VALID_SLIDE_SUFFIXES = (
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif", ".mdsx"
)


# =====================================================================
# GeoJSON
# =====================================================================

def polygonal_parts(geom):
    if geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom

    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms

    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from polygonal_parts(g)


def repair_polygonal(geom):
    if geom.is_empty:
        return geom

    if not geom.is_valid:
        geom = geom.buffer(0)

    parts = list(polygonal_parts(geom))

    if not parts:
        return GeometryCollection()

    return unary_union(parts)


def geojson_features(data):
    if isinstance(data, dict):

        if data.get("type") == "FeatureCollection":
            return data.get("features", [])

        if data.get("type") == "Feature":
            return [data]

        if data.get("type") in {
            "Polygon",
            "MultiPolygon",
            "GeometryCollection",
        }:
            return [
                {
                    "type": "Feature",
                    "geometry": data,
                    "properties": {},
                }
            ]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported GeoJSON structure")


def load_tumor_roi(path: Path):
    """Load all polygonal tumor annotations using the cell-filtering convention."""

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    geoms = []

    for feature in geojson_features(data):

        if not isinstance(feature, dict):
            continue

        geometry = feature.get("geometry")

        if not geometry:
            continue

        try:
            geom = repair_polygonal(shape(geometry))
        except Exception:
            continue

        if geom.is_empty:
            continue

        geoms.append(geom)

    if not geoms:
        raise RuntimeError(
            f"No polygonal tumor ROI found: {path}"
        )

    return repair_polygonal(unary_union(geoms))


def find_tumor_geojson(slide: Path) -> Optional[Path]:

    for suffix in (".geojson", ".json"):

        p = slide.with_suffix(suffix)

        if p.is_file() and p.stat().st_size > 0:
            return p

    return None


# =====================================================================
# Input discovery
# =====================================================================

def collect_slides(
    roots: Sequence[str],
) -> List[Path]:

    out = []

    for raw in roots:

        p = Path(raw)

        if p.is_file():
            if p.suffix.lower() in VALID_SLIDE_SUFFIXES:
                out.append(p.resolve())
            continue

        if not p.is_dir():
            print(f"WARNING: WSI dir not found: {p}")
            continue

        for x in sorted(p.iterdir()):

            if (
                x.is_file()
                and x.suffix.lower()
                in VALID_SLIDE_SUFFIXES
            ):
                out.append(x.resolve())

    return out


def find_assignment_csv(
    slide: Path,
    assignment_dir: Path,
    cell_source: str,
) -> Optional[Path]:

    stem = slide.stem

    direct = (
        assignment_dir
        / f"{stem}_{cell_source}_cell_patch_assignment.csv"
    )

    if direct.is_file():
        return direct

    hits = sorted(
        assignment_dir.rglob(
            f"{stem}_{cell_source}_cell_patch_assignment.csv"
        )
    )

    return hits[0] if hits else None


def find_patch_annotation_csv(
    slide: Path,
    annotation_dir: Path,
) -> Optional[Path]:

    candidates = [
        annotation_dir
        / f"{slide.name}_patch_annotation.csv",

        annotation_dir
        / f"{slide.stem}_patch_annotation.csv",
    ]

    for candidate in candidates:

        if candidate.is_file():
            return candidate

    patterns = [
        f"{slide.name}*_patch_annotation.csv",
        f"{slide.stem}*_patch_annotation.csv",
    ]

    for pattern in patterns:

        hits = sorted(
            annotation_dir.rglob(pattern)
        )

        if hits:
            return hits[0]

    return None


# =====================================================================
# Assignment CSV
# =====================================================================

def truthy(value) -> bool:

    text = str(value or "").strip().lower()

    return text in {
        "1", "true", "yes", "y"
    }


def load_unmatched_cells(
    csv_path: Path,
):
    """Return the total cell count, unmatched cell records, and coordinate offset."""

    unmatched = []

    n_total = 0

    delta_xy = None

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                f"No header: {csv_path}"
            )

        fields = {
            x.strip().lower(): x
            for x in reader.fieldnames
        }

        x_key = fields.get("centroid_x_px")
        y_key = fields.get("centroid_y_px")

        in_key = fields.get("in_tumor_patch")

        dx_key = fields.get("delta_x")
        dy_key = fields.get("delta_y")

        if not x_key or not y_key or not in_key:
            raise ValueError(
                'Assignment CSV is missing '
                "centroid_x_px / centroid_y_px / "
                "in_tumor_patch"
            )

        for row in reader:

            n_total += 1

            if delta_xy is None:

                try:
                    dx = int(
                        float(row.get(dx_key, 0))
                    )
                    dy = int(
                        float(row.get(dy_key, 0))
                    )

                    delta_xy = (dx, dy)

                except Exception:
                    pass

            if truthy(row.get(in_key)):
                continue

            try:
                cx = float(row[x_key])
                cy = float(row[y_key])

            except Exception:
                continue

            if not (
                math.isfinite(cx)
                and math.isfinite(cy)
            ):
                continue

            rec = dict(row)

            rec["_cx"] = cx
            rec["_cy"] = cy

            unmatched.append(rec)

    if delta_xy is None:
        delta_xy = (0, 0)

    return n_total, unmatched, delta_xy


# =====================================================================
# Patch annotation CSV
# =====================================================================

def load_patch_annotation(
    csv_path: Path,
    delta_xy: Tuple[int, int],
):
    """Index annotations by actual read coordinates. Use read_x and read_y when present, or subtract the historical offset from the tile coordinates."""

    patches = {}

    dx, dy = delta_xy

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ValueError(
                f"No header: {csv_path}"
            )

        fields = {
            x.strip().lower(): x
            for x in reader.fieldnames
        }

        tx_key = fields.get("tile_x")
        ty_key = fields.get("tile_y")

        in_key = fields.get("in_annotation")

        rx_key = fields.get("read_x")
        ry_key = fields.get("read_y")

        occ_key = fields.get("occupancy")
        overlap_key = fields.get("overlap_ratio")

        if (
            tx_key is None
            or ty_key is None
            or in_key is None
        ):
            raise ValueError(
                'Patch annotation CSV is missing '
                "tile_x/tile_y/in_annotation"
            )

        for row in reader:

            try:
                tile_x = int(float(row[tx_key]))
                tile_y = int(float(row[ty_key]))

            except Exception:
                continue

            if (
                rx_key
                and ry_key
                and row.get(rx_key, "") != ""
                and row.get(ry_key, "") != ""
            ):

                true_x = int(
                    float(row[rx_key])
                )

                true_y = int(
                    float(row[ry_key])
                )

            else:

                true_x = tile_x - dx
                true_y = tile_y - dy

            occupancy = ""

            if occ_key:
                occupancy = row.get(
                    occ_key,
                    "",
                )

            overlap = ""

            if overlap_key:
                overlap = row.get(
                    overlap_key,
                    "",
                )

            patches[(true_x, true_y)] = {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "true_x": true_x,
                "true_y": true_y,
                "in_annotation": int(
                    truthy(row.get(in_key))
                ),
                "occupancy": occupancy,
                "overlap_ratio": overlap,
            }

    return patches


# =====================================================================
# Rebuild the full tiling grid.
# =====================================================================

def build_full_grid(
    slide_path: Path,
    patch_level: int,
    tile_size: int,
    use_mpp_mode: bool,
    target_mpp: Optional[float],
    roi_margin: int,
    foreground_threshold: Optional[float],
):
    """Use the tiler's ROI, scaling, and coordinate helpers to reconstruct the grid before occupancy filtering."""

    roi_level = (
        0
        if use_mpp_mode
        else int(patch_level)
    )

    loader = base.LoadROId(
        WSIReader(
            backend="OpenSlide"
        ),
        level=roi_level,
        margin=int(roi_margin),
        foreground_threshold=foreground_threshold,
        read_image=False,
    )

    sample = loader({
        "image": str(slide_path),
        "slide_id": slide_path.name,
        "metadata": {},
    })

    origin_xy = (
        int(sample["origin_xy"][0]),
        int(sample["origin_xy"][1]),
    )

    roi_h = int(
        sample["roi_size_hw"][0]
    )
    roi_w = int(
        sample["roi_size_hw"][1]
    )

    fg_threshold = float(
        sample["foreground_threshold"]
    )

    coord_downsample = float(
        sample["scale"]
    )

    effective_mpp = False

    if use_mpp_mode:

        base_mpp = base.read_slide_mpp(
            slide_path
        )

        if (
            base_mpp is not None
            and target_mpp
        ):

            coord_downsample = (
                float(target_mpp)
                / float(base_mpp)
            )

            effective_mpp = True

    if effective_mpp:

        grid_h, grid_w = (
            base.scale_roi_size_hw(
                roi_h,
                roi_w,
                coord_downsample,
            )
        )

    else:

        grid_h, grid_w = (
            roi_h,
            roi_w,
        )

    grid_coords, _ = (
        base.compute_grid_coords(
            grid_h,
            grid_w,
            int(tile_size),
        )
    )

    grid = {}

    for rel_loc in grid_coords:

        loc_l0 = base.rel_to_level0_xy(
            rel_loc,
            coord_downsample,
            origin_xy,
        )

        true_x = int(loc_l0[0])
        true_y = int(loc_l0[1])

        grid[(true_x, true_y)] = (
            int(rel_loc[0]),
            int(rel_loc[1]),
        )

    l0_tile_side = max(
        1,
        int(
            round(
                tile_size
                * coord_downsample
            )
        ),
    )

    return {
        "grid": grid,
        "origin_xy": origin_xy,
        "roi_hw": (roi_h, roi_w),
        "grid_hw": (grid_h, grid_w),
        "coord_downsample": coord_downsample,
        "l0_tile_side": l0_tile_side,
        "fg_threshold": fg_threshold,
        "roi_level": roi_level,
        "effective_mpp": effective_mpp,
    }


# =====================================================================
# Find the theoretical grid tile for a cell.
# =====================================================================

def theoretical_tile(
    cx: float,
    cy: float,
    full_grid: Dict[Tuple[int, int], Tuple[int, int]],
    step: int,
):
    """Find the theoretical tile on a regular level-0 lattice using the minimum grid coordinates as the origin."""

    if not full_grid:
        return None

    xs = [
        p[0]
        for p in full_grid.keys()
    ]

    ys = [
        p[1]
        for p in full_grid.keys()
    ]

    ox = min(xs)
    oy = min(ys)

    tx = (
        ox
        + math.floor(
            (cx - ox) / step
        ) * step
    )

    ty = (
        oy
        + math.floor(
            (cy - oy) / step
        ) * step
    )

    return int(tx), int(ty)


# =====================================================================
# Optional occupancy recomputation
# =====================================================================

def read_tile_at_level(
    slide: OpenSlide,
    rel_xy,
    tile_px: int,
    level: int,
    coord_downsample: float,
    origin_xy,
    roi_hw,
):
    """Read a tile using the same padding as the annotation script so its occupancy can be checked."""

    rel_x = int(rel_xy[0])
    rel_y = int(rel_xy[1])

    roi_h = int(roi_hw[0])
    roi_w = int(roi_hw[1])

    canvas = np.full(
        (tile_px, tile_px, 3),
        255,
        dtype=np.uint8,
    )

    cx0 = max(0, rel_x)
    cy0 = max(0, rel_y)

    cx1 = min(
        roi_w,
        rel_x + tile_px,
    )

    cy1 = min(
        roi_h,
        rel_y + tile_px,
    )

    if cx0 >= cx1 or cy0 >= cy1:
        return np.moveaxis(
            canvas,
            -1,
            0,
        )

    loc_x = int(
        round(
            origin_xy[0]
            + cx0 * coord_downsample
        )
    )

    loc_y = int(
        round(
            origin_xy[1]
            + cy0 * coord_downsample
        )
    )

    region = np.asarray(
        slide.read_region(
            (loc_x, loc_y),
            level,
            (
                cx1 - cx0,
                cy1 - cy0,
            ),
        )
    )[:, :, :3]

    canvas[
        cy0 - rel_y:cy1 - rel_y,
        cx0 - rel_x:cx1 - rel_x,
        :
    ] = region

    return np.moveaxis(
        canvas,
        -1,
        0,
    )


def recompute_occupancy(
    slide: OpenSlide,
    rel_xy,
    grid_info,
    tile_size: int,
):

    tile = read_tile_at_level(
        slide,
        rel_xy,
        tile_size,
        grid_info["roi_level"],
        grid_info["coord_downsample"],
        grid_info["origin_xy"],
        grid_info["grid_hw"],
    )

    fg_mask, _ = base.segment_foreground(
        tile,
        grid_info["fg_threshold"],
    )

    return float(
        fg_mask.mean(
            dtype=np.float32
        )
    )


# =====================================================================
# Visualization helpers
# =====================================================================

CATEGORY_COLORS = {
    "annotation_boundary": (
        255, 40, 40, 235
    ),
    "tissue_boundary": (
        30, 140, 255, 235
    ),
    "grid_roi_boundary": (
        160, 60, 220, 235
    ),
    "unexpected": (
        255, 0, 255, 255
    ),
}


def draw_roi_outline(
    draw,
    roi,
    sx,
    sy,
):

    for poly in polygonal_parts(roi):

        pts = [
            (
                float(x) * sx,
                float(y) * sy,
            )
            for x, y
            in poly.exterior.coords
        ]

        draw.line(
            pts,
            fill=(
                255,
                215,
                0,
                255,
            ),
            width=3,
        )

        for hole in poly.interiors:

            pts = [
                (
                    float(x) * sx,
                    float(y) * sy,
                )
                for x, y
                in hole.coords
            ]

            draw.line(
                pts,
                fill=(
                    255,
                    165,
                    0,
                    255,
                ),
                width=2,
            )


def save_overlay(
    slide_path: Path,
    tumor_roi,
    diagnostics,
    out_path: Path,
    max_thumb: int,
    l0_tile_side: int,
):

    with OpenSlide(str(slide_path)) as slide:

        slide_w, slide_h = (
            slide.dimensions
        )

        thumb = slide.get_thumbnail(
            (
                max_thumb,
                max_thumb,
            )
        ).convert("RGB")

    sx = (
        thumb.width
        / float(slide_w)
    )

    sy = (
        thumb.height
        / float(slide_h)
    )

    overlay = Image.new(
        "RGBA",
        thumb.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(
        overlay
    )

    # Tumor boundary
    draw_roi_outline(
        draw,
        tumor_roi,
        sx,
        sy,
    )

    # Draw patch outlines first.
    for rec in diagnostics:

        category = rec["category"]

        color = CATEGORY_COLORS.get(
            category,
            (255, 255, 255, 220),
        )

        tx = rec["theoretical_true_x"]
        ty = rec["theoretical_true_y"]

        if tx == "" or ty == "":
            continue

        x0 = float(tx) * sx
        y0 = float(ty) * sy

        x1 = (
            float(tx + l0_tile_side)
            * sx
        )

        y1 = (
            float(ty + l0_tile_side)
            * sy
        )

        # Use low-opacity patch outlines.
        box_color = (
            color[0],
            color[1],
            color[2],
            90,
        )

        draw.rectangle(
            (
                x0,
                y0,
                x1,
                y1,
            ),
            outline=box_color,
            width=1,
        )

    # Draw cells on top of the patches.
    for rec in diagnostics:

        category = rec["category"]

        color = CATEGORY_COLORS.get(
            category,
            (255, 255, 255, 255),
        )

        cx = (
            float(rec["centroid_x_px"])
            * sx
        )

        cy = (
            float(rec["centroid_y_px"])
            * sy
        )

        r = 3

        draw.ellipse(
            (
                cx - r,
                cy - r,
                cx + r,
                cy + r,
            ),
            fill=color,
        )

    result = Image.alpha_composite(
        thumb.convert("RGBA"),
        overlay,
    ).convert("RGB")

    # Use English legend labels for portability across system fonts.
    draw2 = ImageDraw.Draw(
        result
    )

    counts = Counter(
        rec["category"]
        for rec in diagnostics
    )

    legend = [
        (
            "Red: annotation boundary",
            "annotation_boundary",
        ),
        (
            "Blue: tissue/foreground boundary",
            "tissue_boundary",
        ),
        (
            "Purple: grid/ROI boundary",
            "grid_roi_boundary",
        ),
        (
            "Magenta: unexpected",
            "unexpected",
        ),
    ]

    x = 12
    y = 12

    for text, category in legend:

        color = CATEGORY_COLORS[
            category
        ][:3]

        n = counts.get(
            category,
            0,
        )

        draw2.rectangle(
            (
                x,
                y + 2,
                x + 12,
                y + 14,
            ),
            fill=color,
        )

        draw2.text(
            (
                x + 18,
                y,
            ),
            f"{text}: {n}",
            fill=(
                255,
                255,
                255,
            ),
            stroke_width=2,
            stroke_fill=(
                0,
                0,
                0,
            ),
        )

        y += 20

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.save(
        out_path,
        format="PNG",
        optimize=True,
    )


# =====================================================================
# Per-case diagnostics
# =====================================================================

DIAGNOSTIC_FIELDS = [
    "cell_id",
    "cell_type",
    "centroid_x_px",
    "centroid_y_px",

    "category",

    "theoretical_true_x",
    "theoretical_true_y",

    "patch_row_exists",
    "patch_in_annotation",

    "tumor_boundary_distance_px",
    "near_tumor_boundary",

    "exact_tumor_overlap_ratio",

    "patch_csv_occupancy",
    "patch_csv_overlap_ratio",

    "recomputed_occupancy",

    "delta_x",
    "delta_y",
]


def process_slide(
    slide_path: Path,
    assignment_csv: Path,
    patch_csv: Path,
    tumor_geojson: Path,
    outdir: Path,
    args,
):

    print("=" * 72)
    print(f"Slide      : {slide_path.name}")
    print(f"Assignment : {assignment_csv}")
    print(f"Patch CSV  : {patch_csv}")
    print(f"Tumor ROI  : {tumor_geojson}")

    (
        n_total,
        unmatched,
        delta_xy,
    ) = load_unmatched_cells(
        assignment_csv
    )

    print(
        f"Cells      : {n_total:,}"
    )

    print(
        f"Unmatched  : {len(unmatched):,}"
    )

    print(
        f"Delta      : {delta_xy}"
    )

    if not unmatched:
        print("No unmatched cells.")
        return

    tumor_roi = load_tumor_roi(
        tumor_geojson
    )

    patch_rows = load_patch_annotation(
        patch_csv,
        delta_xy,
    )

    grid_info = build_full_grid(
        slide_path=slide_path,
        patch_level=args.patch_level,
        tile_size=args.tile_size,
        use_mpp_mode=args.use_mpp_mode,
        target_mpp=args.target_mpp,
        roi_margin=args.roi_margin,
        foreground_threshold=(
            args.foreground_threshold
        ),
    )

    full_grid = grid_info["grid"]

    step = int(
        grid_info["l0_tile_side"]
    )

    print(
        f"Full grid  : {len(full_grid):,} tiles"
    )

    print(
        f"L0 tile    : {step}px"
    )

    diagnostics = []

    # Compute occupancy once per patch.
    occupancy_cache = {}

    slide_obj = OpenSlide(
        str(slide_path)
    )

    try:

        for idx, cell in enumerate(
            unmatched,
            start=1,
        ):

            cx = float(
                cell["_cx"]
            )

            cy = float(
                cell["_cy"]
            )

            tile_xy = theoretical_tile(
                cx,
                cy,
                full_grid,
                step,
            )

            if tile_xy is None:

                category = (
                    "grid_roi_boundary"
                )

                tx = ty = ""

                patch_rec = None

            else:

                tx, ty = tile_xy

                patch_rec = patch_rows.get(
                    (tx, ty)
                )

                if patch_rec is not None:

                    if (
                        patch_rec[
                            "in_annotation"
                        ]
                        == 0
                    ):

                        category = (
                            "annotation_boundary"
                        )

                    else:

                        # An annotated patch should already have accepted this cell.
                        category = (
                            "unexpected"
                        )

                else:

                    if (
                        (tx, ty)
                        in full_grid
                    ):

                        # The annotation tiler applies the foreground occupancy threshold.
                        #
                        # occupancy <= threshold
                        # It skips patches below that threshold.
                        # Skipped patches have no annotation CSV row.
                        category = (
                            "tissue_boundary"
                        )

                    else:

                        category = (
                            "grid_roi_boundary"
                        )

            # ---------------------------------------------------------
            # Measure distance to the tumor boundary.
            # ---------------------------------------------------------

            point = Point(
                cx,
                cy,
            )

            boundary_distance = float(
                tumor_roi.boundary.distance(
                    point
                )
            )

            near_tumor_boundary = int(
                boundary_distance
                <= args.boundary_margin
            )

            # ---------------------------------------------------------
            # Compute exact patch overlap with the tumor polygon using Shapely.
            #
            # Use this measurement for diagnosis.
            # The annotation tiler computes overlap from a raster mask.
            # The original in_annotation field remains the reference.
            # ---------------------------------------------------------

            exact_overlap = ""

            if tx != "":

                tile_geom = box(
                    tx,
                    ty,
                    tx + step,
                    ty + step,
                )

                exact_overlap = float(
                    tile_geom.intersection(
                        tumor_roi
                    ).area
                    / float(step * step)
                )

            # ---------------------------------------------------------
            # occupancy
            # ---------------------------------------------------------

            recomputed_occ = ""

            if (
                args.recompute_occupancy
                and tx != ""
                and (tx, ty) in full_grid
            ):

                key = (tx, ty)

                if key not in occupancy_cache:

                    rel_xy = full_grid[key]

                    occupancy_cache[key] = (
                        recompute_occupancy(
                            slide_obj,
                            rel_xy,
                            grid_info,
                            args.tile_size,
                        )
                    )

                recomputed_occ = (
                    occupancy_cache[key]
                )

            diag = {
                "cell_id": cell.get(
                    "cell_id",
                    idx,
                ),

                "cell_type": cell.get(
                    "cell_type",
                    "",
                ),

                "centroid_x_px": cx,
                "centroid_y_px": cy,

                "category": category,

                "theoretical_true_x": tx,
                "theoretical_true_y": ty,

                "patch_row_exists": (
                    1
                    if patch_rec is not None
                    else 0
                ),

                "patch_in_annotation": (
                    patch_rec[
                        "in_annotation"
                    ]
                    if patch_rec is not None
                    else ""
                ),

                "tumor_boundary_distance_px": (
                    f"{boundary_distance:.4f}"
                ),

                "near_tumor_boundary": (
                    near_tumor_boundary
                ),

                "exact_tumor_overlap_ratio": (
                    f"{exact_overlap:.6f}"
                    if exact_overlap != ""
                    else ""
                ),

                "patch_csv_occupancy": (
                    patch_rec[
                        "occupancy"
                    ]
                    if patch_rec is not None
                    else ""
                ),

                "patch_csv_overlap_ratio": (
                    patch_rec[
                        "overlap_ratio"
                    ]
                    if patch_rec is not None
                    else ""
                ),

                "recomputed_occupancy": (
                    f"{recomputed_occ:.6f}"
                    if recomputed_occ != ""
                    else ""
                ),

                "delta_x": delta_xy[0],
                "delta_y": delta_xy[1],
            }

            diagnostics.append(
                diag
            )

            if idx % 10000 == 0:

                print(
                    f"  diagnosed "
                    f"{idx:,}/"
                    f"{len(unmatched):,}"
                )

    finally:

        slide_obj.close()

    # -------------------------------------------------------------
    # Write diagnostic outputs.
    # -------------------------------------------------------------

    stem = slide_path.stem

    diag_path = (
        outdir
        / f"{stem}_unmatched_diagnostics.csv"
    )

    overlay_path = (
        outdir
        / f"{stem}_unmatched_overlay.png"
    )

    summary_path = (
        outdir
        / f"{stem}_unmatched_summary.csv"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with diag_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=DIAGNOSTIC_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            diagnostics
        )

    counts = Counter(
        rec["category"]
        for rec in diagnostics
    )

    near_count = sum(
        int(
            rec[
                "near_tumor_boundary"
            ]
        )
        for rec in diagnostics
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        fields = [
            "sample_id",
            "n_total_cells",
            "n_unmatched",
            "unmatched_rate",
            "category",
            "n_cells",
            "fraction_of_unmatched",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for category in [
            "annotation_boundary",
            "tissue_boundary",
            "grid_roi_boundary",
            "unexpected",
        ]:

            n = counts.get(
                category,
                0,
            )

            writer.writerow({
                "sample_id": stem,
                "n_total_cells": n_total,
                "n_unmatched": len(
                    diagnostics
                ),
                "unmatched_rate": (
                    len(diagnostics)
                    / n_total
                    if n_total
                    else 0
                ),
                "category": category,
                "n_cells": n,
                "fraction_of_unmatched": (
                    n
                    / len(diagnostics)
                    if diagnostics
                    else 0
                ),
            })

    save_overlay(
        slide_path,
        tumor_roi,
        diagnostics,
        overlay_path,
        args.max_thumb,
        step,
    )

    print()
    print("Diagnosis:")
    print(
        f"  annotation boundary : "
        f"{counts.get('annotation_boundary', 0):,}"
    )
    print(
        f"  tissue boundary     : "
        f"{counts.get('tissue_boundary', 0):,}"
    )
    print(
        f"  grid / ROI boundary : "
        f"{counts.get('grid_roi_boundary', 0):,}"
    )
    print(
        f"  unexpected          : "
        f"{counts.get('unexpected', 0):,}"
    )

    print(
        f"  <= {args.boundary_margin}px "
        f"from tumor boundary: "
        f"{near_count:,} / "
        f"{len(diagnostics):,} "
        f"({100 * near_count / len(diagnostics):.2f}%)"
    )

    print()
    print(
        f"Diagnostic CSV : {diag_path}"
    )
    print(
        f"Overlay PNG    : {overlay_path}"
    )
    print(
        f"Summary CSV    : {summary_path}"
    )


# =====================================================================
# main
# =====================================================================

def build_argparser():

    ap = argparse.ArgumentParser(
        description=(
            "Visualize and diagnose CellViT "
            "cells that were not assigned "
            "to tumor patches."
        )
    )

    ap.add_argument(
        "--wsi-dirs",
        nargs="+",
        default=DEFAULT_WSI_DIRS,
    )

    ap.add_argument(
        "--assignment-dir",
        type=Path,
        default=Path(
            DEFAULT_ASSIGNMENT_DIR
        ),
    )

    ap.add_argument(
        "--patch-annotation-dir",
        type=Path,
        default=Path(
            DEFAULT_PATCH_ANNOTATION_DIR
        ),
    )

    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            DEFAULT_OUTPUT_DIR
        ),
    )

    ap.add_argument(
        "--cell-source",
        choices=[
            "cells",
            "cell_detection",
        ],
        default=DEFAULT_CELL_SOURCE,
    )

    ap.add_argument(
        "--slide",
        default=None,
        help=(
            'Process only the specified slide stem, '
            'for example 1000001.'
        ),
    )

    ap.add_argument(
        "--patch-level",
        type=int,
        default=DEFAULT_PATCH_LEVEL,
    )

    ap.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
    )

    ap.add_argument(
        "--use-mpp-mode",
        action="store_true",
        default=DEFAULT_USE_MPP_MODE,
    )

    ap.add_argument(
        "--target-mpp",
        type=float,
        default=DEFAULT_TARGET_MPP,
    )

    ap.add_argument(
        "--occupancy-threshold",
        type=float,
        default=DEFAULT_OCCUPANCY_THRESHOLD,
    )

    ap.add_argument(
        "--foreground-threshold",
        type=float,
        default=DEFAULT_FOREGROUND_THRESHOLD,
    )

    ap.add_argument(
        "--roi-margin",
        type=int,
        default=DEFAULT_ROI_MARGIN,
    )

    ap.add_argument(
        "--annotation-overlap-threshold",
        type=float,
        default=(
            DEFAULT_ANNOTATION_OVERLAP_THRESHOLD
        ),
    )

    ap.add_argument(
        "--boundary-margin",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN_PX,
        help=(
            'Maximum distance in level-0 pixels for a cell '
            'to count as near the tumor boundary.'
        ),
    )

    ap.add_argument(
        "--max-thumb",
        type=int,
        default=DEFAULT_MAX_THUMB,
    )

    ap.add_argument(
        "--no-recompute-occupancy",
        dest="recompute_occupancy",
        action="store_false",
        default=True,
        help=(
            'Skip rereading unmatched patches '
            'to verify foreground occupancy.'
        ),
    )

    return ap


def main():

    args = (
        build_argparser()
        .parse_args()
    )

    slides = collect_slides(
        args.wsi_dirs
    )

    if args.slide:

        wanted = (
            Path(args.slide)
            .stem
            .lower()
        )

        slides = [
            slide
            for slide in slides
            if slide.stem.lower()
            == wanted
        ]

    if not slides:
        raise SystemExit(
            "No WSI found."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for slide in slides:

        assignment_csv = (
            find_assignment_csv(
                slide,
                args.assignment_dir,
                args.cell_source,
            )
        )

        patch_csv = (
            find_patch_annotation_csv(
                slide,
                args.patch_annotation_dir,
            )
        )

        tumor_geojson = (
            find_tumor_geojson(
                slide
            )
        )

        missing = []

        if assignment_csv is None:
            missing.append(
                "assignment CSV"
            )

        if patch_csv is None:
            missing.append(
                "patch annotation CSV"
            )

        if tumor_geojson is None:
            missing.append(
                "tumor GeoJSON"
            )

        if missing:

            print(
                f"SKIP {slide.name}: "
                f"missing "
                f"{', '.join(missing)}"
            )

            n_skip += 1

            continue

        try:

            process_slide(
                slide,
                assignment_csv,
                patch_csv,
                tumor_geojson,
                args.output_dir,
                args,
            )

            n_ok += 1

        except Exception as exc:

            n_fail += 1

            print(
                f"ERROR {slide.name}: "
                f"{exc}",
                file=sys.stderr,
            )

    print()
    print("=" * 72)

    print(
        f"Done. "
        f"processed={n_ok} "
        f"skipped={n_skip} "
        f"failed={n_fail}"
    )


if __name__ == "__main__":
    main()
