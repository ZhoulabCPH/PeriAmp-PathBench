"""Render Integrated Gradients results saved in NPZ files. Override the module defaults using command-line arguments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from integrated_gradients import (
    build_attribution_overlay,
    magnitude_asinh_norm,
    magnitude_color_limits,
    make_attribution_canvas,
    save_attribution_heatmap,
    save_attribution_histogram,
    save_top_patches_figure,
    signed_asinh_norm,
    signed_color_norm,
    openslide,
)
from overlay_pdf import save_overlay_outputs
from tumor_region import (
    add_tumor_region_arguments,
    prepare_tumor_display,
    tumor_filter_kwargs,
)
from heatmap_common import (
    SUPPORTED_SLIDE_EXTS,
    category_path,
    find_slide_for_stem,
    infer_case_id_from_npz as infer_case_id,
    iter_npz,
    npz_scalar as scalar,
    npz_text_list as text_list,
    output_name_from_npz as output_name,
)


# ============================================================
# Default settings
# ============================================================

DEFAULT_NPZ = Path(
    r"outputs/interpretability/integrated_gradients/attributions"
)
DEFAULT_OUTDIR = Path(r"outputs/interpretability/integrated_gradients/IG_plots")
DEFAULT_SLIDE: Optional[Path] = None
DEFAULT_SLIDE_DIR: Optional[Path] = r"data/slides"       # Match slides by case stem and extension.
DEFAULT_SLIDE_EXTS = SUPPORTED_SLIDE_EXTS

DEFAULT_TILE_SIZE = 256
DEFAULT_DOWNSAMPLE = 16
DEFAULT_CMAP = "RdBu_r"
DEFAULT_TOP_K = 20
DEFAULT_CLIP_LO = 1
DEFAULT_CLIP_HI = 97
# Color scaling for signed integrated gradients
# asinh uses symmetric percentile limits with magnitude compression.
# Equal magnitudes have equal intensity and the weaker tail remains visible.
# symmetric uses linear limits of plus or minus the magnitude percentile.
# per_side scales each sign independently so intensities differ across signs.
DEFAULT_SIGNED_SCALE = "symmetric"
SIGNED_SCALE_MODES = ("asinh", "symmetric", "per_side")
DEFAULT_ALPHA = 0.50
DEFAULT_SMOOTH_SIGMA = 2.0
DEFAULT_DARKEN_BG = 1.0

# Tumor annotation filtering
DEFAULT_TUMOR_FILTER = False
DEFAULT_ANNOTATION_DIR: Optional[Path] = Path(
    r"outputs/interpretability/tumor_area_output"
)
DEFAULT_ANNOTATION_CSV: Optional[Path] = None
DEFAULT_TUMOR_DISPLAY_MODE = "zero_outside"

# ============================================================


def plot_one(npz_path: Path, args: argparse.Namespace) -> None:
    with np.load(npz_path, allow_pickle=True) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    if "patch_attributions" not in data or "coords" not in data:
        raise KeyError(f"{npz_path} is missing patch_attributions or coords.")

    values = np.asarray(data["patch_attributions"], dtype=np.float64).reshape(-1)
    coords = np.asarray(data["coords"])
    if coords.ndim != 2 or coords.shape[0] != values.shape[0] or coords.shape[1] < 2:
        raise ValueError(f"{npz_path} has incompatible attribution and coordinate shapes: {values.shape}, {coords.shape}")

    case_id = infer_case_id(npz_path, data)
    disp = prepare_tumor_display(
        coords,
        values,
        case_id,
        display_mode=args.tumor_display_mode,
        clip_lo=args.clip_lo,
        clip_hi=args.clip_hi,
        **tumor_filter_kwargs(args),
    )
    plot_coords = disp.plot_coords
    plot_values = disp.plot_values
    top_coords = disp.tumor_coords
    top_values = disp.tumor_values
    if disp.filter_info["enabled"] and disp.tumor_values.size == 0:
        raise ValueError(f"{npz_path} has no patches after filtering.")
    if plot_values.size == 0:
        raise ValueError(f"{npz_path} has no patches after filtering.")

    # Estimate both maps' limits using the tumor attribution magnitudes.
    # Use the same upper magnitude for the signed and absolute maps.
    mag_lo, mag_hi = magnitude_color_limits(top_values, args.clip_lo, args.clip_hi)
    signed_norm = None
    abs_norm = None
    signed_vmin, signed_vmax = -mag_hi, mag_hi
    if args.signed_scale == "per_side":
        signed_norm = signed_color_norm(top_values, args.clip_lo, args.clip_hi)
        if signed_norm is None:
            print('[color scale] per_side requires both attribution signs. Using symmetric limits.')
        else:
            signed_vmin, signed_vmax = signed_norm.vmin, signed_norm.vmax
    elif args.signed_scale == "asinh":
        # Share the same asinh linear width between the signed and absolute maps.
        signed_norm = signed_asinh_norm(top_values, args.clip_lo, args.clip_hi)
        abs_norm = magnitude_asinh_norm(top_values, args.clip_lo, args.clip_hi)
        signed_vmin, signed_vmax = signed_norm.vmin, signed_norm.vmax
    signed_desc = f"signed({args.signed_scale})=[{signed_vmin:.4g}, {signed_vmax:.4g}]"
    if abs_norm is not None:
        signed_desc += f" linear_width={signed_norm.linear_width:.4g}"
        abs_desc = f"|IG|=[0, {mag_hi:.4g}](asinh)"
    else:
        abs_desc = f"|IG|=[{mag_lo:.4g}, {mag_hi:.4g}]"
    print(
        f"[color scale] |IG| P{args.clip_lo:g}={mag_lo:.4g}  P{args.clip_hi:g}={mag_hi:.4g}"
        f" → {signed_desc}, {abs_desc}"
    )

    class_names = text_list(data.get("class_names")) or ["class_0"]
    pred_class = int(scalar(data, "pred_class", 0) or 0)
    target_class = int(scalar(data, "target_class", pred_class) or pred_class)
    pred_class = min(max(pred_class, 0), len(class_names) - 1)
    target_class = min(max(target_class, 0), len(class_names) - 1)

    outroot = Path(args.outdir)
    output_prefix = output_name(npz_path, case_id)
    print(f"[plotting] {npz_path} -> {outroot} (grouped by output type)")

    canvas, mask, origin_xy, _ = make_attribution_canvas(
        values=plot_values, coords=plot_coords,
        tile_size=args.tile_size, downsample=args.downsample,
    )
    save_attribution_heatmap(
        canvas,
        category_path(outroot, "ig_attribution_heatmap", output_prefix, "png"),
        cmap=args.cmap,
        symmetric=True, title=f"IG Attribution - Target: {class_names[target_class]}",
        vmin=signed_vmin, vmax=signed_vmax, norm=signed_norm,
    )
    abs_canvas, _, _, _ = make_attribution_canvas(
        values=np.abs(plot_values), coords=plot_coords,
        tile_size=args.tile_size, downsample=args.downsample,
    )
    save_attribution_heatmap(
        abs_canvas,
        category_path(
            outroot, "ig_attribution_abs_heatmap", output_prefix, "png"
        ),
        cmap="magma",
        symmetric=False, title=f"|IG Attribution| - Target: {class_names[target_class]}",
        vmin=mag_lo, vmax=mag_hi, norm=abs_norm,
    )
    save_top_patches_figure(
        top_values, top_coords, class_names, target_class,
        category_path(outroot, "ig_top_patches", output_prefix, "png"),
        top_k=args.top_k,
    )
    save_attribution_histogram(
        top_values,
        category_path(outroot, "ig_attribution_histogram", output_prefix, "png"),
        class_names,
        target_class, clip_lo=args.clip_lo, clip_hi=args.clip_hi,
        vmin=signed_vmin, vmax=signed_vmax,
    )

    slide_path = args.slide
    if slide_path is None and args.slide_dir is not None:
        slide_path = find_slide_for_stem(Path(args.slide_dir), case_id)
        if slide_path is None:
            print(f"[overlay] No slide found for {case_id}. Skipping the overlay.")
    if slide_path:
        if openslide is None:
            print('[warning] Install openslide-python to generate WSI overlays.')
        else:
            layers = build_attribution_overlay(
                slide_path=str(slide_path), attr_canvas=canvas, mask=mask,
                origin_xy=origin_xy, downsample=args.downsample, cmap=args.cmap,
                symmetric=True, alpha=args.alpha, smooth_sigma=args.smooth_sigma,
                darken_bg=args.darken_bg,
                vmin=signed_vmin, vmax=signed_vmax, norm=signed_norm,
            )
            from PIL import Image

            Image.fromarray((layers.thumbnail * 255).astype(np.uint8)).save(
                category_path(outroot, "wsi_thumbnail", output_prefix, "png")
            )
            save_overlay_outputs(
                layers,
                category_path(
                    outroot, "ig_attribution_overlay", output_prefix, "png"
                ),
                category_path(
                    outroot, "ig_attribution_overlay", output_prefix, "pdf"
                ),
                alpha=args.alpha,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description='Render heatmaps from saved Integrated Gradients NPZ files.')
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ,
                        help='Input NPZ file or directory')
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help='Output root. Results are grouped by output type.',
    )
    parser.add_argument("--slide", type=Path, default=DEFAULT_SLIDE)
    parser.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR,
                        help='Directory used to find slides by case stem and extension')
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--cmap", type=str, default=DEFAULT_CMAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--clip-lo", type=float, default=DEFAULT_CLIP_LO)
    parser.add_argument("--clip-hi", type=float, default=DEFAULT_CLIP_HI)
    parser.add_argument(
        "--signed-scale",
        choices=list(SIGNED_SCALE_MODES),
        default=DEFAULT_SIGNED_SCALE,
        help=(
            'Signed IG color scale. '
            'asinh uses symmetric magnitude percentiles and compresses large magnitudes. '
            'Equal magnitudes have equal intensities across signs. '
            'symmetric uses the same limits with a linear mapping. '
            'per_side uses separate percentile limits for the positive and negative tails.'
        ),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--darken-bg", type=float, default=DEFAULT_DARKEN_BG)
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
        raise ValueError('Provide --npz or set DEFAULT_NPZ.')
    paths = iter_npz(args.npz)
    if not paths:
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")
    for path in paths:
        try:
            plot_one(path, args)
        except Exception as exc:
            print(f"[error] {path.name}: {exc}")


if __name__ == "__main__":
    main()
