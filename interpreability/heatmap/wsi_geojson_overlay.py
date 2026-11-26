#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render level-0 GeoJSON coordinates on a WSI thumbnail and write PNG and PDF overlays.
Supports individual geometries, features, feature collections, and top-level lists. Feature collections can be read incrementally with ijson.
"""

import argparse
import json
from pathlib import Path

import ijson
from PIL import Image, ImageDraw

WSI_PATH = r"data/slides/slide_001.svs"

GEOJSON_PATH = r"outputs/interpretability/attention/attention_geojson/slide_001-GigaPath-ABMIL-Fold0-attentions_attention_vis.geojson"

OUTPUT_DIR = r"outputs/interpretability/wsi_overlay"


# Thumbnail long edge in pixels
MAX_THUMB = 8000

# Polygon and line width
LINE_WIDTH = 2

# Point radius
POINT_RADIUS = 3

# PNG compression level from 0 to 9
PNG_COMPRESS_LEVEL = 1

# PDF resolution metadata
PDF_DPI = 150


# ======================================================================
# Default colors
# ======================================================================

PANNUKE_COLORS = {
    1: (255, 0, 0),
    2: (34, 221, 77),
    3: (35, 92, 236),
    4: (254, 255, 0),
    5: (255, 159, 68),
}

NAME_COLORS = {
    "neoplastic": (255, 0, 0),
    "tumor": (255, 0, 0),
    "tumour": (255, 0, 0),

    "inflammatory": (34, 221, 77),

    "connective": (35, 92, 236),
    "stromal": (35, 92, 236),
    "stroma": (35, 92, 236),

    "dead": (254, 255, 0),
    "necrosis": (254, 255, 0),

    "epithelial": (255, 159, 68),
}

DEFAULT_COLOR = (0, 255, 255)


# ======================================================================
# WSI
# ======================================================================

def open_slide(path):
    """Open the slide with TiffSlide when possible, falling back to OpenSlide."""
    try:
        from tiffslide import TiffSlide
        slide = TiffSlide(str(path))
        return slide, "TiffSlide"

    except Exception as tiff_error:
        try:
            import openslide
            slide = openslide.OpenSlide(str(path))
            return slide, "OpenSlide"

        except Exception as openslide_error:
            raise RuntimeError(
                'Could not open the whole-slide image.\n\n'
                f"TiffSlide error:\n{tiff_error}\n\n"
                f"OpenSlide error:\n{openslide_error}"
            )


# ======================================================================
# Slide and coordinate helpers
# ======================================================================

def first_non_whitespace_byte(path):
    with open(path, "rb") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return b""
            if ch not in b" \t\r\n":
                return ch


def read_top_level_type(path):
    """Read the top-level GeoJSON type without loading the entire collection."""
    try:
        with open(path, "rb") as f:
            for value in ijson.items(f, "type"):
                return str(value)
    except Exception:
        pass
    return None


def xy(coord):
    """Convert a coordinate pair to floats."""
    if not isinstance(coord, (list, tuple)):
        return None
    if len(coord) < 2:
        return None

    try:
        x = float(coord[0])
        y = float(coord[1])
    except (TypeError, ValueError):
        return None

    if x != x or y != y:
        return None

    return x, y


# ======================================================================
# Color parsing
# ======================================================================

def parse_color(value):
    """Parse an RGB triplet, hex color, or comma-separated RGB string."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(
                max(0, min(255, int(float(v))))
                for v in value[:3]
            )
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()

        if text.startswith("#") and len(text) >= 7:
            try:
                return (
                    int(text[1:3], 16),
                    int(text[3:5], 16),
                    int(text[5:7], 16),
                )
            except ValueError:
                return None

        parts = [x.strip() for x in text.split(",")]

        if len(parts) >= 3:
            try:
                return tuple(
                    max(0, min(255, int(float(v))))
                    for v in parts[:3]
                )
            except ValueError:
                return None

    return None


def get_feature_style(properties):
    """Resolve style properties from QuPath, CellViT, or generic GeoJSON."""
    if not isinstance(properties, dict):
        properties = {}

    classification = properties.get("classification")
    if not isinstance(classification, dict):
        classification = {}

    name = (
        classification.get("name")
        or properties.get("cell_type")
        or properties.get("class")
        or properties.get("label")
        or properties.get("name")
        or properties.get("type")
        or "Unknown"
    )
    name = str(name)

    color = (
        parse_color(classification.get("color"))
        or parse_color(properties.get("color"))
        or parse_color(properties.get("stroke"))
        or parse_color(properties.get("fill"))
    )

    if color is not None:
        return name, color

    type_id = (
        classification.get("type")
        if classification.get("type") is not None
        else properties.get("type_id")
    )

    try:
        type_id = int(type_id)
    except (TypeError, ValueError):
        type_id = None

    if type_id in PANNUKE_COLORS:
        return name, PANNUKE_COLORS[type_id]

    mapped = NAME_COLORS.get(name.strip().lower())
    if mapped is not None:
        return name, mapped

    return name, DEFAULT_COLOR


# ======================================================================
# Stream GeoJSON objects.
# ======================================================================

def iter_geojson_objects(path):
    """Yield GeoJSON features and geometries. Stream feature collections and top-level lists with ijson."""
    first = first_non_whitespace_byte(path)

    if first == b"[":
        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
        return

    if first != b"{":
        raise RuntimeError(
            "A GeoJSON file must start with '{' or '['."
        )

    top_type = read_top_level_type(path)

    if top_type == "FeatureCollection":
        with open(path, "rb") as f:
            for feature in ijson.items(f, "features.item"):
                if isinstance(feature, dict):
                    yield feature
        return

    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(
            'The top-level GeoJSON value is not a valid object.'
        )

    yield obj


# ======================================================================
# Map slide coordinates onto the thumbnail.
# ======================================================================

def scale_xy(coord, sx, sy):
    point = xy(coord)
    if point is None:
        return None

    x, y = point
    return x * sx, y * sy


def scale_coordinate_list(coords, sx, sy):
    points = []

    if not isinstance(coords, list):
        return points

    for coord in coords:
        point = scale_xy(coord, sx, sy)
        if point is not None:
            points.append(point)

    return points


# ======================================================================
# Draw GeoJSON geometries.
# ======================================================================

def draw_geometry(
    draw,
    geometry,
    sx,
    sy,
    color,
    line_width,
    point_radius,
):
    """Draw a GeoJSON geometry and return the number of rendered primitives."""
    if not isinstance(geometry, dict):
        return 0

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    rgba = tuple(color[:3]) + (230,)

    if geometry_type == "Point":
        point = scale_xy(coordinates, sx, sy)
        if point is None:
            return 0

        x, y = point
        r = point_radius

        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=rgba,
        )
        return 1

    if geometry_type == "MultiPoint":
        if not isinstance(coordinates, list):
            return 0

        count = 0
        for coord in coordinates:
            point = scale_xy(coord, sx, sy)
            if point is None:
                continue

            x, y = point
            r = point_radius

            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=rgba,
            )
            count += 1

        return count

    if geometry_type == "LineString":
        points = scale_coordinate_list(
            coordinates,
            sx,
            sy,
        )
        if len(points) < 2:
            return 0

        draw.line(
            points,
            fill=rgba,
            width=line_width,
            joint="curve",
        )
        return 1

    if geometry_type == "MultiLineString":
        if not isinstance(coordinates, list):
            return 0

        count = 0
        for line in coordinates:
            points = scale_coordinate_list(
                line,
                sx,
                sy,
            )

            if len(points) < 2:
                continue

            draw.line(
                points,
                fill=rgba,
                width=line_width,
                joint="curve",
            )
            count += 1

        return count

    if geometry_type == "Polygon":
        if not isinstance(coordinates, list):
            return 0

        count = 0

        for ring in coordinates:
            points = scale_coordinate_list(
                ring,
                sx,
                sy,
            )

            if len(points) < 2:
                continue

            if points[0] != points[-1]:
                points.append(points[0])

            draw.line(
                points,
                fill=rgba,
                width=line_width,
                joint="curve",
            )
            count += 1

        return 1 if count else 0

    if geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list):
            return 0

        polygon_count = 0

        for polygon in coordinates:
            if not isinstance(polygon, list):
                continue

            has_valid_ring = False

            for ring in polygon:
                points = scale_coordinate_list(
                    ring,
                    sx,
                    sy,
                )

                if len(points) < 2:
                    continue

                if points[0] != points[-1]:
                    points.append(points[0])

                draw.line(
                    points,
                    fill=rgba,
                    width=line_width,
                    joint="curve",
                )

                has_valid_ring = True

            if has_valid_ring:
                polygon_count += 1

        return polygon_count

    if geometry_type == "GeometryCollection":
        geometries = geometry.get("geometries")

        if not isinstance(geometries, list):
            return 0

        count = 0

        for sub_geometry in geometries:
            count += draw_geometry(
                draw=draw,
                geometry=sub_geometry,
                sx=sx,
                sy=sy,
                color=color,
                line_width=line_width,
                point_radius=point_radius,
            )

        return count

    return 0


# ======================================================================
# Feature / Geometry
# ======================================================================

def draw_geojson_object(
    draw,
    obj,
    sx,
    sy,
    line_width,
    point_radius,
):
    if not isinstance(obj, dict):
        return 0

    obj_type = obj.get("type")

    if obj_type == "Feature":
        _, color = get_feature_style(
            obj.get("properties")
        )

        return draw_geometry(
            draw=draw,
            geometry=obj.get("geometry"),
            sx=sx,
            sy=sy,
            color=color,
            line_width=line_width,
            point_radius=point_radius,
        )

    valid_geometry_types = {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    }

    if obj_type in valid_geometry_types:
        _, color = get_feature_style(
            obj.get("properties")
        )

        return draw_geometry(
            draw=draw,
            geometry=obj,
            sx=sx,
            sy=sy,
            color=color,
            line_width=line_width,
            point_radius=point_radius,
        )

    if isinstance(obj.get("geometry"), dict):
        _, color = get_feature_style(
            obj.get("properties")
        )

        return draw_geometry(
            draw=draw,
            geometry=obj.get("geometry"),
            sx=sx,
            sy=sy,
            color=color,
            line_width=line_width,
            point_radius=point_radius,
        )

    return 0


# ======================================================================
# Overlay generation
# ======================================================================

def create_overlay(
    slide_path,
    geojson_path,
    output_dir,
    max_thumb=8000,
    line_width=2,
    point_radius=3,
):
    slide_path = Path(slide_path).expanduser().resolve()
    geojson_path = Path(geojson_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not slide_path.is_file():
        raise FileNotFoundError(
            f"WSI not found: {slide_path}"
        )

    if not geojson_path.is_file():
        raise FileNotFoundError(
            f"GeoJSON not found: {geojson_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = (
        f"{slide_path.stem}_"
        f"{geojson_path.stem}_overlay"
    )

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    print("=" * 70)
    print("WSI + GeoJSON -> PNG + PDF")
    print("=" * 70)

    print(f"WSI     : {slide_path}")
    print(f"GeoJSON : {geojson_path}")
    print(f"Output  : {output_dir}")

    print()
    print("Opening WSI...")

    slide, backend = open_slide(slide_path)

    try:
        slide_width, slide_height = map(
            int,
            slide.dimensions,
        )

        print(f"Backend     : {backend}")
        print(
            f"WSI size    : "
            f"{slide_width:,} x {slide_height:,}"
        )

        print()
        print(
            f"Creating thumbnail "
            f"(max side = {max_thumb}px)..."
        )

        thumbnail = slide.get_thumbnail(
            (max_thumb, max_thumb)
        ).convert("RGB")

        thumb_width, thumb_height = thumbnail.size

        print(
            f"Thumbnail   : "
            f"{thumb_width:,} x {thumb_height:,}"
        )

        sx = thumb_width / slide_width
        sy = thumb_height / slide_height

        print(
            f"Scale X     : {sx:.8f}"
        )
        print(
            f"Scale Y     : {sy:.8f}"
        )

        overlay = Image.new(
            "RGBA",
            thumbnail.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(overlay)

        print()
        print("Drawing GeoJSON...")

        object_count = 0
        primitive_count = 0

        for obj in iter_geojson_objects(
            geojson_path
        ):
            object_count += 1

            primitive_count += draw_geojson_object(
                draw=draw,
                obj=obj,
                sx=sx,
                sy=sy,
                line_width=line_width,
                point_radius=point_radius,
            )

            if object_count % 100000 == 0:
                print(
                    f"  processed "
                    f"{object_count:,} objects..."
                )

        result = Image.alpha_composite(
            thumbnail.convert("RGBA"),
            overlay,
        ).convert("RGB")

        print()
        print("Saving PNG...")

        result.save(
            png_path,
            format="PNG",
            optimize=False,
            compress_level=PNG_COMPRESS_LEVEL,
        )

        print("Saving PDF...")

        result.save(
            pdf_path,
            format="PDF",
            resolution=PDF_DPI,
        )

        print()
        print("=" * 70)
        print("DONE")
        print("=" * 70)

        print(
            f"GeoJSON objects : "
            f"{object_count:,}"
        )

        print(
            f"Drawn geometries: "
            f"{primitive_count:,}"
        )

        print()
        print(f"PNG : {png_path}")
        print(f"PDF : {pdf_path}")

        print("=" * 70)

        return png_path, pdf_path

    finally:
        try:
            slide.close()
        except Exception:
            pass


# ======================================================================
# Command line
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay arbitrary GeoJSON annotations "
            "onto a whole-slide thumbnail "
            "and save PNG + PDF."
        )
    )

    parser.add_argument(
        "--slide",
        default=WSI_PATH,
        help="WSI path",
    )

    parser.add_argument(
        "--geojson",
        default=GEOJSON_PATH,
        help="GeoJSON path",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--max-thumb",
        type=int,
        default=MAX_THUMB,
        help="Maximum thumbnail side",
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=LINE_WIDTH,
        help="Polygon/line width",
    )

    parser.add_argument(
        "--point-radius",
        type=float,
        default=POINT_RADIUS,
        help="Point radius",
    )

    args = parser.parse_args()

    create_overlay(
        slide_path=args.slide,
        geojson_path=args.geojson,
        output_dir=args.output_dir,
        max_thumb=args.max_thumb,
        line_width=args.line_width,
        point_radius=args.point_radius,
    )


if __name__ == "__main__":
    main()
