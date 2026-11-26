# Render saved ABMIL attention NPZ files.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from calculate_attention import (
    build_attention_dot_overlay,
    build_attribution_overlay,
    find_slide_for_case,
    normalize_attention,
    save_attention_dot_overlay,
    save_attention_dotmap,
    save_attribution_heatmap,
    save_top_patches_figure,
    openslide,
)
from integrated_gradients import make_attribution_canvas
from overlay_pdf import save_overlay_outputs
from tumor_region import (
    add_tumor_region_arguments,
    prepare_tumor_display,
    tumor_filter_kwargs,
)
from heatmap_common import (
    category_path,
    infer_case_id_from_npz as _case_id,
    iter_npz,
    npz_scalar as scalar,
    npz_text_list as text_list,
    output_name_from_npz as output_name,
)


# ============================================================
# Shared plot settings
# ============================================================

# Input and output paths

# Single NPZ file or a directory of NPZ files
DEFAULT_NPZ = Path(
    r"outputs/interpretability/attention/attentions"
)

# Plot output directory
DEFAULT_OUTDIR = Path(
    r"outputs/interpretability/attention/attention_plots"
)

# Provide one slide or let the script look under DEFAULT_SLIDE_DIR.
DEFAULT_SLIDE: Optional[Path] = None

# Original slide directory for overlays
DEFAULT_SLIDE_DIR: Optional[Path] = Path(
    r"data/slides"
)

# Supported slide extensions
DEFAULT_SLIDE_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")


# Values to plot

# Supported NPZ fields
# attention contains raw attention.
# attention_vis contains values prepared for display.
# patch_attributions contains per-tile attribution values.
DEFAULT_VALUE_TYPE = "attention_vis"

# Normalize raw attention using the configured mode.
# none / minmax / rank / log
DEFAULT_ATTN_NORM = "minmax"


# Common plotting settings

# Tile side length in level-0 pixels
DEFAULT_TILE_SIZE = 256

# Downsample factor for low-resolution plots
DEFAULT_DOWNSAMPLE = 16

# Matplotlib colormap
DEFAULT_CMAP = "viridis"

# Number of patches to show in the top-attention plot
DEFAULT_TOP_K = 20

# Lower color-limit percentile
DEFAULT_CLIP_LO = 1

# Upper color-limit percentile
DEFAULT_CLIP_HI = 97


# WSI heatmap overlay settings

# Heatmap opacity over the slide
DEFAULT_ALPHA = 0.55

# Gaussian smoothing strength
DEFAULT_SMOOTH_SIGMA = 2.0

# Background brightness. Values below 1 darken the slide.
DEFAULT_DARKEN_BG = 1.0


# Dot map settings

# Optionally generate a standalone attention dot map.
DEFAULT_DOTMAP_ENABLED = True

# Optionally overlay attention dots on the WSI.
DEFAULT_DOT_OVERLAY_ENABLED = True

# Dot size as a fraction of the patch footprint
DEFAULT_DOTMAP_FILL_RATIO = 0.82

# Dot opacity over the slide
DEFAULT_DOT_OVERLAY_ALPHA = 0.78


# Tumor-region filtering
# Enable tumor-region filtering.
DEFAULT_TUMOR_FILTER = False

# Directory containing tumor annotation CSVs
DEFAULT_ANNOTATION_DIR: Optional[Path] = Path(
    r"outputs/interpretability/tumor_area_output"
)
# Optional override for a single annotation CSV.
DEFAULT_ANNOTATION_CSV: Optional[Path] = None

# Display options when tumor filtering is enabled.
# filter draws only tiles inside the tumor annotation.
# zero_outside draws all tissue and assigns zero to other tiles.
DEFAULT_TUMOR_DISPLAY_MODE = "zero_outside"

# ============================================================


def _values(data: dict, value_type: str) -> np.ndarray:
    """Read the selected tile-level values from the NPZ file."""
    key = {
        "attention": "attention",
        "attention_vis": "attention_vis",
        "patch_attributions": "patch_attributions",
    }[value_type]

    if key not in data:
        raise KeyError(f"NPZ is missing {key}")

    return np.asarray(data[key], dtype=np.float64).reshape(-1)


def _prepare_display_values(
    values: np.ndarray,
    value_type: str,
    attn_norm: str,
) -> np.ndarray:
    """Normalize raw attention for display and use saved display values or patch attributions directly."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if value_type == "attention":
        return normalize_attention(values, attn_norm)

    return values


def plot_one(npz_path: Path, args: argparse.Namespace) -> None:
    # ========================================================
    # Read the saved attention arrays.
    # ========================================================
    with np.load(npz_path, allow_pickle=True) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    if "coords" not in data:
        raise KeyError(f"{npz_path} is missing coords.")

    values = _values(data, args.value_type)
    coords = np.asarray(data["coords"])

    if (
        coords.ndim != 2
        or coords.shape[0] != values.shape[0]
        or coords.shape[1] < 2
    ):
        raise ValueError(
            f"{npz_path} has incompatible value and coordinate shapes: "
            f"{values.shape}, {coords.shape}"
        )

    case_id = _case_id(npz_path, data)

    # ========================================================
    # Apply tumor filtering and the display mode.
    # ========================================================
    disp = prepare_tumor_display(
        coords,
        values,
        case_id,
        display_mode=args.tumor_display_mode,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        value_transform=lambda v: _prepare_display_values(
            v, args.value_type, args.attn_norm
        ),
        **tumor_filter_kwargs(args),
    )
    plot_coords = disp.plot_coords
    plot_values = disp.plot_values
    top_coords = disp.tumor_coords
    top_values = disp.tumor_values
    plot_clip_lo = disp.clip_lo
    plot_clip_hi = disp.clip_hi

    if disp.filter_info["enabled"] and disp.tumor_values.size == 0:
        raise ValueError(f"{npz_path} has no patches inside the tumor annotation.")

    if plot_values.size == 0:
        raise ValueError(f"{npz_path} has no patches to plot.")

    # ========================================================
    # Prepare output paths and class metadata.
    # ========================================================
    class_names = text_list(data.get("class_names")) or ["class_0"]

    pred_class = int(scalar(data, "pred_class", 0) or 0)
    pred_class = min(max(pred_class, 0), len(class_names) - 1)

    outroot = Path(args.outdir)
    output_prefix = output_name(npz_path, case_id)

    print(f"[plotting] {npz_path} -> {outroot} (grouped by output type)")

    title = f"ABMIL Attention - {case_id}"

    # ========================================================
    # Build a two-dimensional tile canvas.
    # ========================================================
    canvas, mask, origin_xy, _ = make_attribution_canvas(
        values=plot_values,
        coords=plot_coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
    )

    # ========================================================
    # Draw the heatmap.
    # ========================================================
    save_attribution_heatmap(
        canvas,
        category_path(outroot, "attn_heatmap", output_prefix, "png"),
        cmap=args.cmap,
        symmetric=False,
        title=title,
        clip_lo=plot_clip_lo,
        clip_hi=plot_clip_hi,
    )

    # ========================================================
    # 6. Top-K patch
    # Select top patches from the tumor subset when filtering is enabled.
    # ========================================================
    save_top_patches_figure(
        top_values,
        top_coords,
        class_names,
        pred_class,
        category_path(outroot, "attn_top_patches", output_prefix, "png"),
        top_k=args.top_k,
    )

    # ========================================================
    # Optional standalone dot map
    # ========================================================
    if args.dotmap_enabled:
        save_attention_dotmap(
            out_path_png=category_path(
                outroot, "attn_slide_dotmap", output_prefix, "png"
            ),
            attn_vis=plot_values,
            coords=plot_coords,
            cmap=args.cmap,
            title=title,
            tile_size=args.tile_size,
            fill_ratio=args.dotmap_fill_ratio,
            clip_lo=plot_clip_lo,
            clip_hi=plot_clip_hi,
        )

    # ========================================================
    # Locate the slide for overlays.
    # ========================================================
    slide_path = args.slide

    if slide_path is None and args.slide_dir is not None:
        slide_path = find_slide_for_case(
            case_id,
            None,
            args.slide_dir,
            tuple(args.slide_exts),
        )

    # Keep standalone plots when the slide is unavailable.
    if slide_path is None:
        print(f"[overlay] case={case_id} has no matching WSI. Skipping the overlay.")
        return

    if openslide is None:
        print('[warning] Install openslide-python to generate WSI overlays.')
        return

    # ========================================================
    # Build the continuous attention overlay.
    # ========================================================
    layers = build_attribution_overlay(
        slide_path=str(slide_path),
        attr_canvas=canvas,
        mask=mask,
        origin_xy=origin_xy,
        downsample=args.downsample,
        cmap=args.cmap,
        symmetric=False,
        alpha=args.alpha,
        smooth_sigma=args.smooth_sigma,
        darken_bg=args.darken_bg,
        clip_lo=plot_clip_lo,
        clip_hi=plot_clip_hi,
    )

    Image.fromarray(
        (layers.thumbnail * 255).astype(np.uint8)
    ).save(
        category_path(outroot, "wsi_thumbnail", output_prefix, "png")
    )

    save_overlay_outputs(
        layers,
        category_path(outroot, "attn_overlay", output_prefix, "png"),
        category_path(outroot, "attn_overlay", output_prefix, "pdf"),
        alpha=args.alpha,
    )

    # ========================================================
    # 10. attention dot overlay + WSI
    # ========================================================
    if args.dot_overlay_enabled:
        dot_layers = build_attention_dot_overlay(
            slide_path=str(slide_path),
            attn_vis=plot_values,
            coords=plot_coords,
            cmap=args.cmap,
            tile_size=args.tile_size,
            downsample=args.downsample,
            alpha=args.dot_overlay_alpha,
            fill_ratio=args.dotmap_fill_ratio,
            darken_bg=args.darken_bg,
            clip_lo=plot_clip_lo,
            clip_hi=plot_clip_hi,
        )

        save_attention_dot_overlay(
            category_path(
                outroot, "attn_dot_overlay", output_prefix, "png"
            ),
            dot_layers,
            category_path(
                outroot, "attn_dot_overlay", output_prefix, "pdf"
            ),
            alpha=args.dot_overlay_alpha,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render heatmaps from saved attention NPZ files.'
    )

    parser.add_argument(
        "--npz",
        type=Path,
        default=DEFAULT_NPZ,
        help='Input NPZ file or directory',
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help='Output root. Results are grouped by output type.',
    )

    parser.add_argument(
        "--slide",
        type=Path,
        default=DEFAULT_SLIDE,
    )

    parser.add_argument(
        "--slide-dir",
        type=Path,
        default=DEFAULT_SLIDE_DIR,
    )

    parser.add_argument(
        "--slide-exts",
        nargs="*",
        default=list(DEFAULT_SLIDE_EXTS),
    )

    parser.add_argument(
        "--value-type",
        choices=[
            "attention",
            "attention_vis",
            "patch_attributions",
        ],
        default=DEFAULT_VALUE_TYPE,
    )

    parser.add_argument(
        "--attn-norm",
        choices=[
            "none",
            "minmax",
            "rank",
            "log",
        ],
        default=DEFAULT_ATTN_NORM,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_TILE_SIZE,
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default=DEFAULT_CMAP,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--clip-lo",
        type=float,
        default=DEFAULT_CLIP_LO,
    )

    parser.add_argument(
        "--clip-hi",
        type=float,
        default=DEFAULT_CLIP_HI,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA,
    )

    parser.add_argument(
        "--darken-bg",
        type=float,
        default=DEFAULT_DARKEN_BG,
    )

    parser.add_argument(
        "--dotmap-fill-ratio",
        type=float,
        default=DEFAULT_DOTMAP_FILL_RATIO,
    )

    parser.add_argument(
        "--dot-overlay-alpha",
        type=float,
        default=DEFAULT_DOT_OVERLAY_ALPHA,
    )

    parser.add_argument(
        "--no-dotmap",
        dest="dotmap_enabled",
        action="store_false",
        default=DEFAULT_DOTMAP_ENABLED,
    )

    parser.add_argument(
        "--no-dot-overlay",
        dest="dot_overlay_enabled",
        action="store_false",
        default=DEFAULT_DOT_OVERLAY_ENABLED,
    )

    # Add the shared tumor-region options.
    # --annotation-dir
    # --annotation-csv
    # --tumor-filter
    # --no-tumor-filter
    # --tumor-display-mode
    add_tumor_region_arguments(
        parser,
        annotation_dir_default=DEFAULT_ANNOTATION_DIR,
        annotation_csv_default=DEFAULT_ANNOTATION_CSV,
        tumor_filter_default=DEFAULT_TUMOR_FILTER,
        display_mode_default=DEFAULT_TUMOR_DISPLAY_MODE,
    )
    parser.set_defaults(tumor_filter=DEFAULT_TUMOR_FILTER)

    args = parser.parse_args()

    if args.npz is None:
        raise ValueError(
            'Provide --npz or set DEFAULT_NPZ.'
        )

    paths = iter_npz(args.npz)

    if not paths:
        raise FileNotFoundError(
            f"NPZ file not found: {args.npz}"
        )

    for path in paths:
        try:
            plot_one(path, args)
        except Exception as exc:
            print(f"[error] {path.name}: {exc}")


if __name__ == "__main__":
    main()
