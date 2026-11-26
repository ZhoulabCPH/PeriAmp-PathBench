"""Compute patch-level Integrated Gradients for trained MIL models and write attribution NPZ files. Generate heatmaps and WSI overlays separately with plot_ig_from_npz.py."""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, TwoSlopeNorm
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

# Add MIL_pipeline to the import path for model classes.
def _find_mil_pipeline_dir(start: Path) -> Path:
    """Locate MIL_pipeline relative to the script."""
    for p in [start, *start.parents]:
        candidate = p / "MIL_pipeline"
        if (candidate / "model").is_dir():
            return candidate
        if p.name == "MIL_pipeline" and (p / "model").is_dir():
            return p
    raise FileNotFoundError(
        f"MIL_pipeline/model was not found while searching upward from {start}."
        'Check that the checkout contains MIL_pipeline/model.'
    )


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_MIL_PIPELINE_DIR = _find_mil_pipeline_dir(_SCRIPT_DIR)
if str(_MIL_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_MIL_PIPELINE_DIR))

from overlay_pdf import OverlayLayers, build_scalar_overlay_layers, save_overlay_outputs  # noqa: E402
# ============================================================
# Default settings
# Command-line arguments override these defaults.
# ============================================================

# --config-json / --model-class / --model-params / --weights / --fold
DEFAULT_CONFIG_JSON = (
    r"outputs/mil/UNI/wikg_run/config.json"
)
DEFAULT_MODEL_CLASS = "model.ABMIL.ABMIL"
DEFAULT_MODEL_PARAMS = None
DEFAULT_WEIGHTS = None
DEFAULT_FOLD = 0

# --h5 / --class-names
DEFAULT_H5 = r"data/features/UNI/slide_001.h5"
DEFAULT_CLASS_NAMES = ["Cholangiocarcinoma", "Ampullary", "Duodenal", "Pancreatic"]

# --target-class / --n-steps / --ig-batch-size / --baseline
DEFAULT_TARGET_CLASS = None
DEFAULT_N_STEPS = 100
DEFAULT_IG_BATCH_SIZE = 1
DEFAULT_BASELINE = "zero"                      # {zero, mean, random}

# --outdir
DEFAULT_OUTDIR = r"outputs/interpretability/integrated_gradients"

# ============================================================


# Utility functions

def load_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        coords = f["coords"][:]
    return features, coords


def parse_model_class(class_path: str):
    """Import the configured MIL model class dynamically."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def load_mil_model(model_class_path: str, model_params: dict, weights_path: str, device: torch.device):
    """Build the MIL model and restore its weights."""
    model_cls = parse_model_class(model_class_path)
    model = model_cls(**model_params).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[model] {model_class_path} ← {weights_path}")
    return model


def load_config_and_model(config_path: str, fold_idx: int, device: torch.device):
    """Restore the model from the training config and return the model and config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model_class_path = config["model"]["class"]
    model_params = config["model"]["params"]

    save_dir = str(Path(config_path).parent)
    weights_path = os.path.join(save_dir, f"fold_{fold_idx}_best.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    model = load_mil_model(model_class_path, model_params, weights_path, device)
    return model, config


# Integrated Gradients computation

def integrated_gradients(
    model,
    features: torch.Tensor,
    mask: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    n_steps: int = 50,
    batch_size: int = 10,
    device: torch.device = None,
):
    """Compute per-tile attribution and a completeness delta for the target class. Interpolate from the supplied or zero baseline to the feature tensor, running the interpolation points in batches to bound device memory."""
    if device is None:
        device = next(model.parameters()).device

    features = features.to(device)
    mask = mask.to(device)

    if baseline is None:
        baseline = torch.zeros_like(features)
    else:
        baseline = baseline.to(device)

    diff = features - baseline  # [1, N, D]

    alphas = torch.linspace(0, 1, n_steps + 1, device=device)

    # Give trapezoid endpoints weight 0.5 and interior points weight 1.
    trap_weights = torch.ones(n_steps + 1, device=device)
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5

    scaled_grads = torch.zeros_like(features)  # [1, N, D]

    model.eval()

    for start in range(0, n_steps + 1, batch_size):
        end = min(start + batch_size, n_steps + 1)
        alpha_batch = alphas[start:end]  # [bs]
        w_batch = trap_weights[start:end]  # [bs]
        bs = len(alpha_batch)

        interp_inputs = baseline + alpha_batch.view(bs, 1, 1) * diff
        interp_inputs = interp_inputs.detach().requires_grad_(True)

        mask_batch = mask.expand(bs, -1)

        logits = model(interp_inputs, mask_batch)  # [bs, C]
        target_scores = logits[:, target_class].sum()

        target_scores.backward()

        # Accumulate gradients using trapezoidal weights.
        scaled_grads += (interp_inputs.grad * w_batch.view(bs, 1, 1)).sum(dim=0, keepdim=True)

        model.zero_grad()

    # Divide the weighted gradient sum by the number of steps.
    avg_grads = scaled_grads / n_steps  # [1, N, D]

    # Multiply the average gradient by input minus baseline.
    attributions = avg_grads * diff  # [1, N, D]

    # Check completeness against the target output difference.
    with torch.no_grad():
        f_input = model(features, mask)[:, target_class].item()
        f_baseline = model(baseline, mask)[:, target_class].item()
    convergence_delta = attributions.sum().item() - (f_input - f_baseline)

    # Sum over the feature dimension to obtain one score per tile.
    patch_attributions = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()  # [N]

    patch_attributions = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()  # [N]
    return patch_attributions, convergence_delta


# Attribution visualizations

def resolve_clip_percentiles(
    clip_lo: float = None,
    clip_hi: float = None,
    percentile_clip: float = None,
):
    """Resolve visual clipping bounds. The legacy percentile_clip value p maps to (p, 100 - p)."""
    if percentile_clip is not None:
        if clip_lo is None:
            clip_lo = percentile_clip
        if clip_hi is None:
            clip_hi = 100.0 - percentile_clip

    if clip_lo is None:
        clip_lo = 0.5
    if clip_hi is None:
        clip_hi = 99.5

    clip_lo = float(clip_lo)
    clip_hi = float(clip_hi)
    if not (0.0 <= clip_lo < clip_hi <= 100.0):
        raise ValueError(f"Invalid clipping percentiles: clip_lo={clip_lo}, clip_hi={clip_hi}")
    return clip_lo, clip_hi


def magnitude_color_limits(
    values,
    clip_lo: float = None,
    clip_hi: float = None,
) -> Tuple[float, float]:
    """Derive shared clipping limits from absolute attributions. Signed plots use symmetric limits and absolute plots use the nonnegative magnitude interval."""
    clip_lo, clip_hi = resolve_clip_percentiles(clip_lo, clip_hi)
    mag = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return 0.0, 1e-12

    mag_lo = float(np.percentile(mag, clip_lo))
    mag_hi = float(np.percentile(mag, clip_hi))
    if mag_hi <= mag_lo:
        # Use actual extrema when the percentile interval collapses.
        mag_lo, mag_hi = float(mag.min()), float(mag.max())
    if mag_hi <= mag_lo:
        mag_hi = mag_lo + 1e-12
    return mag_lo, mag_hi


def signed_color_norm(
    values,
    clip_lo: float = None,
    clip_hi: float = None,
):
    """Scale positive and negative tails separately around zero. Equal magnitudes can have different intensities across signs. Return None if either side has no usable percentile."""
    clip_lo, clip_hi = resolve_clip_percentiles(clip_lo, clip_hi)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    pos = v[v > 0]
    neg = -v[v < 0]
    if pos.size == 0 or neg.size == 0:
        return None

    pos_hi = float(np.percentile(pos, clip_hi))
    neg_hi = float(np.percentile(neg, clip_hi))
    if not (pos_hi > 0.0 and neg_hi > 0.0):
        return None
    return TwoSlopeNorm(vcenter=0.0, vmin=-neg_hi, vmax=pos_hi)


def _asinh_linear_width(
    values,
    mag_lo: float,
    mag_hi: float,
    linear_width: Optional[float] = None,
) -> float:
    """Choose the positive half-width of the linear region for asinh scaling, defaulting to the lower magnitude limit."""
    width = float(linear_width) if linear_width is not None else float(mag_lo)
    if not np.isfinite(width) or width <= 0.0:
        mag = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        mag = mag[np.isfinite(mag) & (mag > 0.0)]
        width = float(mag.min()) if mag.size else mag_hi * 1e-3
    return float(np.clip(width, mag_hi * 1e-6, mag_hi))


def signed_asinh_norm(
    values,
    clip_lo: float = None,
    clip_hi: float = None,
    linear_width: Optional[float] = None,
):
    """Apply symmetric asinh scaling with a linear region near zero and logarithmic compression farther out. Equal positive and negative magnitudes have mirrored intensities."""
    mag_lo, mag_hi = magnitude_color_limits(values, clip_lo, clip_hi)
    width = _asinh_linear_width(values, mag_lo, mag_hi, linear_width)
    return AsinhNorm(linear_width=width, vmin=-mag_hi, vmax=mag_hi)


def magnitude_asinh_norm(
    values,
    clip_lo: float = None,
    clip_hi: float = None,
    linear_width: Optional[float] = None,
):
    """Apply the signed map's magnitude limits and linear width to the absolute-value map."""
    mag_lo, mag_hi = magnitude_color_limits(values, clip_lo, clip_hi)
    width = _asinh_linear_width(values, mag_lo, mag_hi, linear_width)
    return AsinhNorm(linear_width=width, vmin=0.0, vmax=mag_hi)


def make_attribution_canvas(
    values: np.ndarray,
    coords: np.ndarray,
    tile_size: int = 256,
    downsample: int = 16,
):
    """Rasterize tile-level attribution values on a spatial canvas."""
    coords = coords.astype(np.int64)
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    max_x, max_y = coords[:, 0].max(), coords[:, 1].max()

    tile_draw = max(1, tile_size // downsample)
    width = int((max_x - min_x) // downsample + tile_draw)
    height = int((max_y - min_y) // downsample + tile_draw)

    canvas = np.full((height, width), np.nan, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    for v, (x, y) in zip(values, coords):
        x0 = int((x - min_x) // downsample)
        y0 = int((y - min_y) // downsample)
        x1 = min(width, x0 + tile_draw)
        y1 = min(height, y0 + tile_draw)
        canvas[y0:y1, x0:x1] = v
        mask[y0:y1, x0:x1] = True

    return canvas, mask, (min_x, min_y), (width, height)


def save_attribution_heatmap(
    canvas: np.ndarray,
    out_path: Path,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    colorbar: bool = True,
    title: str = "",
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
    percentile_clip: float = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm=None,
):
    """Save an attribution heatmap using the supplied norm or color limits. Infer limits from clipping percentiles when they are omitted."""
    valid = canvas[~np.isnan(canvas)]
    if len(valid) == 0:
        print('[warning] Heatmap contains no finite values. Skipping.')
        return

    clip_lo, clip_hi = resolve_clip_percentiles(clip_lo, clip_hi, percentile_clip)
    if norm is None:
        vmin = np.percentile(valid, clip_lo) if vmin is None else float(vmin)
        vmax = np.percentile(valid, clip_hi) if vmax is None else float(vmax)
        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="white")

    plt.figure(figsize=(12, 10))
    if norm is None:
        im = plt.imshow(canvas, cmap=cm, vmin=vmin, vmax=vmax)
    else:
        im = plt.imshow(canvas, cmap=cm, norm=norm)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=14)
    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Attribution Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"[saved] Attribution heatmap: {out_path}")


def build_attribution_overlay(
    slide_path: str,
    attr_canvas: np.ndarray,
    mask: np.ndarray,
    origin_xy,
    downsample: int,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    alpha: float = 0.55,
    smooth_sigma: float = 2.0,
    darken_bg: float = 0.8,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
    percentile_clip: float = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm=None,
) -> OverlayLayers:
    """Overlay attribution on a WSI thumbnail and prepare separate layers for PDF export."""
    clip_lo, clip_hi = resolve_clip_percentiles(clip_lo, clip_hi, percentile_clip)
    return build_scalar_overlay_layers(
        slide_path=slide_path,
        attr_canvas=attr_canvas,
        mask=mask,
        origin_xy=origin_xy,
        downsample=downsample,
        cmap=cmap,
        symmetric=symmetric,
        alpha=alpha,
        smooth_sigma=smooth_sigma,
        darken_bg=darken_bg,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )


def save_top_patches_figure(
    patch_attributions: np.ndarray,
    coords: np.ndarray,
    class_names: list,
    target_class: int,
    out_path: Path,
    top_k: int = 20,
):
    """Plot the locations of the most positive and most negative tiles."""
    sorted_idx = np.argsort(patch_attributions)
    top_pos_idx = sorted_idx[-top_k:][::-1]
    top_neg_idx = sorted_idx[:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(coords[:, 0], coords[:, 1], c="lightgray", s=2, alpha=0.3)
    sc0 = axes[0].scatter(
        coords[top_pos_idx, 0], coords[top_pos_idx, 1],
        c=patch_attributions[top_pos_idx], cmap="Reds", s=30, edgecolors="black", linewidths=0.5,
    )
    axes[0].set_title(f"Top-{top_k} Positive Attribution\n(Target: {class_names[target_class]})", fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    plt.colorbar(sc0, ax=axes[0], fraction=0.046)

    axes[1].scatter(coords[:, 0], coords[:, 1], c="lightgray", s=2, alpha=0.3)
    sc1 = axes[1].scatter(
        coords[top_neg_idx, 0], coords[top_neg_idx, 1],
        c=patch_attributions[top_neg_idx], cmap="Blues_r", s=30, edgecolors="black", linewidths=0.5,
    )
    axes[1].set_title(f"Top-{top_k} Negative Attribution\n(Target: {class_names[target_class]})", fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] Top-k patch locations: {out_path}")


def save_attribution_histogram(
    patch_attributions: np.ndarray,
    out_path: Path,
    class_names: list,
    target_class: int,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot signed and absolute attribution distributions with the heatmap's color limits and saturation points. The plots leave the attribution arrays intact."""
    vals = np.asarray(patch_attributions, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        print('[warning] No attribution values to plot.')
        return

    # -----------------------------
    # Compute summary statistics for the figure title.
    # -----------------------------
    mean_val = vals.mean()
    std_val = vals.std()
    max_abs = np.max(np.abs(vals))

    neg_vals = vals[vals < 0]
    pos_vals = vals[vals > 0]
    abs_vals = np.abs(vals)
    abs_nonzero = abs_vals[abs_vals > 0]

    # The display bounds apply only to the plot.
    clip_lo, clip_hi = resolve_clip_percentiles(clip_lo, clip_hi)
    mag_lo, mag_hi = magnitude_color_limits(vals, clip_lo, clip_hi)

    # Use the limits of the signed heatmap by default.
    x_min = -mag_hi if vmin is None else float(vmin)
    x_max = mag_hi if vmax is None else float(vmax)
    if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
        x_abs_max = max(max_abs, 1e-12)
        x_min, x_max = -x_abs_max, x_abs_max

    vals_clip = vals[(vals >= x_min) & (vals <= x_max)]
    if vals_clip.size == 0:
        vals_clip = vals.copy()
    neg_clip = vals_clip[vals_clip < 0]
    pos_clip = vals_clip[vals_clip > 0]

    # -----------------------------
    # Create the figure.
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1, ax2 = axes

    # =========================================================
    # Left panel: signed attribution distribution
    # =========================================================
    n_bins_signed = 80
    bins_signed = np.linspace(x_min, x_max, n_bins_signed + 1)

    if neg_clip.size > 0:
        ax1.hist(
            neg_clip,
            bins=bins_signed,
            color="#4C84D3",
            edgecolor="white",
            alpha=0.9,
            label=f"Negative (n={neg_vals.size})",
        )
    if pos_clip.size > 0:
        ax1.hist(
            pos_clip,
            bins=bins_signed,
            color="#E36A54",
            edgecolor="white",
            alpha=0.9,
            label=f"Positive (n={pos_vals.size})",
        )

    ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax1.axvline(
        mean_val,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean={mean_val:.2e}",
    )
    # Mark the saturation limits used by the heatmap.
    ax1.axvline(
        x_min,
        color="dimgray",
        linewidth=1.2,
        label=f"Heatmap scale=[{x_min:.2e}, {x_max:.2e}]",
    )
    ax1.axvline(x_max, color="dimgray", linewidth=1.2)

    ax1.set_xlim(x_min, x_max)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlabel("Attribution Score", fontsize=12)
    ax1.set_ylabel("Patch Count (symlog)", fontsize=12)
    n_sat = int(np.count_nonzero((vals < x_min) | (vals > x_max)))
    ax1.set_title(
        f"IG Attribution Distribution\n"
        f"Target: {class_names[target_class]}   |   clip@[{clip_lo}, {clip_hi}]%"
        f"   |   saturated {n_sat}/{vals.size}",
        fontsize=13,
    )
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()

    # =========================================================
    # Right panel: absolute attribution distribution
    # =========================================================
    if abs_nonzero.size > 0:
        p50 = np.percentile(abs_nonzero, 50)
        p90 = np.percentile(abs_nonzero, 90)
        p99 = np.percentile(abs_nonzero, 99)

        xmin_mag = max(abs_nonzero.min(), 1e-12)
        xmax_mag = max(abs_nonzero.max(), xmin_mag * 10)

        # Use logarithmic bins for values concentrated near zero.
        n_bins_mag = 80
        bins_mag = np.logspace(np.log10(xmin_mag), np.log10(xmax_mag), n_bins_mag + 1)

        ax2.hist(
            abs_nonzero,
            bins=bins_mag,
            color="#7569D6",
            edgecolor="white",
            alpha=0.9,
        )

        ax2.axvline(p50, color="limegreen", linestyle="--", linewidth=1.2, label=f"P50(|x|)={p50:.2e}")
        ax2.axvline(p90, color="orange", linestyle="--", linewidth=1.2, label=f"P90(|x|)={p90:.2e}")
        ax2.axvline(p99, color="red", linestyle="--", linewidth=1.2, label=f"P99(|x|)={p99:.2e}")
        # Mark the magnitude heatmap's upper color limit.
        ax2.axvline(
            mag_hi,
            color="dimgray",
            linewidth=1.2,
            label=f"|IG| heatmap vmax={mag_hi:.2e}",
        )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("|Attribution Score| (log)", fontsize=12)
        ax2.set_ylabel("Patch Count (log)", fontsize=12)
        ax2.set_title(
            "IG Attribution Magnitude Distribution\n"
            f"Mean={mean_val:.2e}, Std={std_val:.2e}, Max|x|={max_abs:.2e}",
            fontsize=13,
        )
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend()
    else:
        ax2.text(
            0.5, 0.5,
            "All attribution magnitudes are zero.",
            ha="center", va="center", fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.set_title(
            "IG Attribution Magnitude Distribution\n"
            f"Mean={mean_val:.2e}, Std={std_val:.2e}, Max|x|={max_abs:.2e}",
            fontsize=13,
        )
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] Attribution histogram: {out_path}")


# Command-line entry point

def main():
    parser = argparse.ArgumentParser(
        description='Compute Integrated Gradients and save per-patch attribution in NPZ format.'
    )

    # Model arguments
    group_model = parser.add_argument_group('Model configuration')
    group_model.add_argument(
        "--config-json", type=str, default=DEFAULT_CONFIG_JSON,
        help='Training config.json used to restore the model architecture and settings',
    )
    group_model.add_argument(
        "--model-class", type=str, default=DEFAULT_MODEL_CLASS,
        help='MIL model class, for example model.ABMIL.ABMIL. Used without --config-json.',
    )
    group_model.add_argument(
        "--model-params", type=str, default=DEFAULT_MODEL_PARAMS,
        help='JSON object of model parameters. Used without --config-json.',
    )
    group_model.add_argument(
        "--weights", type=str, default=DEFAULT_WEIGHTS,
        help='PyTorch model checkpoint. Used without --config-json.',
    )
    group_model.add_argument("--fold", type=int, default=DEFAULT_FOLD, help='Best checkpoint fold index when using --config-json')

    # Data arguments
    group_data = parser.add_argument_group('Data configuration')
    group_data.add_argument("--h5", type=str, default=DEFAULT_H5, help='HDF5 feature bag to analyze')
    group_data.add_argument(
        "--class-names", type=str, nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help='Class names',
    )
    # Integrated Gradients arguments
    group_ig = parser.add_argument_group('Integrated Gradients options')
    group_ig.add_argument("--target-class", type=int, default=DEFAULT_TARGET_CLASS, help='Target class index. Defaults to the model prediction.')
    group_ig.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help='Number of Integrated Gradients interpolation steps')
    group_ig.add_argument("--ig-batch-size", type=int, default=DEFAULT_IG_BATCH_SIZE, help='Interpolation points per batch. A smaller batch uses less GPU memory.')
    group_ig.add_argument(
        "--baseline", type=str, default=DEFAULT_BASELINE,
        choices=["zero", "mean", "random"],
        help='Baseline: zero vector, mean feature vector, or random noise',
    )

    # Output arguments
    group_output = parser.add_argument_group('Output options')
    group_output.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help='NPZ output directory. Defaults to a subdirectory named for the HDF5 file.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load the trained model.
    if args.config_json:
        model, config = load_config_and_model(args.config_json, args.fold, device)
        class_names = config.get("class_names", args.class_names)
    else:
        if not args.weights or not args.model_params:
            parser.error('Provide both --weights and --model-params when --config-json is omitted.')
        model_params = json.loads(args.model_params)
        model = load_mil_model(args.model_class, model_params, args.weights, device)
        class_names = args.class_names

    n_classes = len(class_names)

    # Load all patches from the HDF5 bag.
    features_np, coords_np = load_h5(args.h5)
    n_patches, feat_dim = features_np.shape
    print(f"[data] Loaded {args.h5}")
    print(f"  Patches: {n_patches}, feature width: {feat_dim}")

    features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, D]
    mask = torch.ones(1, n_patches, dtype=torch.bool).to(device)

    # Predict the slide class.
    with torch.no_grad():
        logits = model(features, mask)  # [1, C]
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_class = int(np.argmax(probs))

    print(f"\n[prediction]")
    for i, name in enumerate(class_names):
        marker = " ← predicted" if i == pred_class else ""
        print(f"  {name}: {probs[i]:.4f}{marker}")

    target_class = args.target_class if args.target_class is not None else pred_class
    print(f"\n[IG target class] {class_names[target_class]} (idx={target_class})")

    # Prepare the attribution baseline.
    if args.baseline == "zero":
        baseline = torch.zeros_like(features)
    elif args.baseline == "mean":
        mean_feat = features.mean(dim=1, keepdim=True).expand_as(features)
        baseline = mean_feat
    elif args.baseline == "random":
        baseline = torch.randn_like(features) * features.std()
    else:
        baseline = torch.zeros_like(features)

    # Compute Integrated Gradients.
    print(f"\n[IG] Computing attributions with steps={args.n_steps}, batch_size={args.ig_batch_size}, baseline={args.baseline}")
    patch_attributions, convergence_delta = integrated_gradients(
        model=model,
        features=features,
        mask=mask,
        target_class=target_class,
        baseline=baseline,
        n_steps=args.n_steps,
        batch_size=args.ig_batch_size,
        device=device,
    )
    print(f"[IG] Completeness delta = {convergence_delta:.6f}")

    # Summarize attributions.
    print(f"\n[attribution statistics]")
    print(f"  Mean:   {patch_attributions.mean():.6f}")
    print(f"  Std:    {patch_attributions.std():.6f}")
    print(f"  Min:    {patch_attributions.min():.6f}")
    print(f"  Max:    {patch_attributions.max():.6f}")
    print(f"  |Pos|:  {(patch_attributions > 0).sum()} patches")
    print(f"  |Neg|:  {(patch_attributions < 0).sum()} patches")

    # Prepare the output directory.
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        h5_stem = Path(args.h5).stem
        outdir = _SCRIPT_DIR / "output_ig" / h5_stem
    outdir.mkdir(parents=True, exist_ok=True)

    # Save attributions as NPZ.
    npz_path = outdir / "attributions.npz"
    np.savez_compressed(
        npz_path,
        patch_attributions=patch_attributions,
        coords=coords_np,
        case_id=Path(args.h5).stem,
        probs=probs,
        pred_class=pred_class,
        target_class=target_class,
        class_names=class_names,
        convergence_delta=convergence_delta,
    )
    print(f"[saved] Attribution data: {npz_path}")

    print('[output] Use plot_ig_from_npz.py to render the saved attribution.')
    print(f"[output directory] {outdir}")


if __name__ == "__main__":
    main()
