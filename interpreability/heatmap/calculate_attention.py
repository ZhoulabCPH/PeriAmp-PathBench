"""
Extract per-tile ABMIL attention into NPZ files.
Load the trained model from --config-json and select weights with --fold. Supply an HDF5 bag or a directory with --h5. Use --average-folds to average attention across folds. Plot the resulting NPZ files with plot_attention_from_npz.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse model loading, canvases, overlays, and top-k plots from integrated_gradients.py.
from integrated_gradients import (   # noqa: E402
    load_h5,
    load_config_and_model,
    load_mil_model,
    make_attribution_canvas,
    save_attribution_heatmap,
    build_attribution_overlay,
    save_top_patches_figure,
)
from overlay_pdf import (  # noqa: E402
    OverlayLayers,
    layers_from_dot_overlay,
    load_wsi_thumbnail,
    save_overlay_outputs,
)
from heatmap_common import iter_h5  # noqa: E402
try:
    import openslide
except ImportError:
    openslide = None


# ============================================================
# Default settings
# ============================================================

DEFAULT_CONFIG_JSON = Path(
    r"outputs/mil/gigapath/abmil_run/config.json"
)
DEFAULT_FOLD = 0
DEFAULT_AVERAGE_FOLDS = False          # Average attention over the available folds when enabled.
DEFAULT_N_FOLDS = 5                    # Number of fold indices to try when averaging

# Supply all three model options when no training config is used.
DEFAULT_MODEL_CLASS = "model.ABMIL.ABMIL"
DEFAULT_MODEL_PARAMS: Optional[dict] = None
DEFAULT_WEIGHTS: Optional[Path] = None

DEFAULT_H5 = Path(r"data/features/gigapath/internal/tile_h5/slide_001.h5")  # Single HDF5 feature bag or directory
DEFAULT_CLASS_NAMES = ["Cholangiocarcinoma", "Ampullary", "Duodenal", "Pancreatic"]
DEFAULT_OUTDIR = Path(r"outputs/interpretability/attention")

# Normalization affects attention_vis only. NPZ files also preserve raw attention.
# none uses raw softmax attention, typically of order 1/N.
# minmax rescales values linearly to [0, 1].
# rank rescales ranks to [0, 1] to reduce outlier effects.
# log applies log1p before min-max scaling.
DEFAULT_ATTN_NORM = "minmax"           # {none, minmax, rank, log}

# ============================================================


ATTN_NORM_MODES = ("none", "minmax", "rank", "log")


# Attention extraction

@torch.no_grad()
def compute_abmil_attention(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Reproduce ABMIL's attention operation for features shaped (1, N, D) and an optional (1, N) mask. Return N attention values, applying softmax when model.is_norm is enabled."""
    required = ("_fc1", "attention_U", "attention_V", "attention_weight")
    if not all(hasattr(model, name) for name in required):
        # Fall back to a hook on the attention_weight layer.
        return _compute_attention_via_hook(model, features, mask)

    x = features
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    x = model._fc1(x)
    a_u = model.attention_U(x)
    a_v = model.attention_V(x)
    a = model.attention_weight(a_u * a_v).transpose(1, 2)  # [1, 1, N]
    if getattr(model, "is_norm", True):
        a = torch.softmax(a, dim=-1)
    return a.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def _compute_attention_via_hook(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> np.ndarray:
    """Capture attention_weight with a forward hook and apply softmax for compatible ABMIL variants."""
    captured = {}

    target = None
    for name, module in model.named_modules():
        if name.endswith("attention_weight"):
            target = module
            break
    if target is None:
        raise AttributeError(
            'Cannot extract attention. This ABMIL variant has no attention_weight layer.'
        )

    def _hook(_m, _inp, out):
        captured["a"] = out.detach()

    handle = target.register_forward_hook(_hook)
    try:
        model(features, mask)
    finally:
        handle.remove()

    a = captured["a"]               # [1, N, 1]
    a = a.transpose(1, 2)           # [1, 1, N]
    a = torch.softmax(a, dim=-1)
    return a.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float64)


def normalize_attention(attn: np.ndarray, mode: str) -> np.ndarray:
    """Return a normalized copy of the attention values for display."""
    mode = (mode or "none").lower()
    if mode not in ATTN_NORM_MODES:
        raise ValueError(f"Unknown attention normalization mode: {mode}. Choose from {ATTN_NORM_MODES}")

    a = np.asarray(attn, dtype=np.float64).copy()
    if a.size == 0 or mode == "none":
        return a

    if mode == "minmax":
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    if mode == "log":
        a = np.log1p(np.clip(a - a.min(), 0, None))
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    if mode == "rank":
        n = a.size
        if n <= 1:
            return np.zeros_like(a)
        order = np.argsort(a, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / (n - 1)

    raise ValueError(f"Unknown attention normalization mode: {mode}")


# Model loading and fold averaging

def load_models(args: argparse.Namespace, device: torch.device):
    """Load the selected fold and its class names."""
    if args.config_json:
        model, config = load_config_and_model(str(args.config_json), args.fold, device)
        class_names = config.get("class_names", args.class_names)
        return [model], class_names

    if not args.weights or not args.model_params:
        raise ValueError('Provide both --weights and --model-params when --config-json is omitted.')
    params = args.model_params
    if isinstance(params, str):
        import json
        params = json.loads(params)
    model = load_mil_model(args.model_class, params, str(args.weights), device)
    return [model], args.class_names


def load_average_fold_models(
    args: argparse.Namespace,
    device: torch.device,
) -> Optional[List[torch.nn.Module]]:
    """Load all fold models for cross-fold attention averaging."""
    if not args.average_folds:
        return None
    if not args.config_json:
        print('Skipping FoldAvg output. Provide --config-json to discover fold weights.')
        return None

    models = []
    for k in range(args.n_folds):
        try:
            m, _config = load_config_and_model(str(args.config_json), k, device)
            models.append(m)
        except FileNotFoundError:
            print(f"[skip] fold_{k}_best.pth was not found.")
    if not models:
        raise FileNotFoundError('No fold checkpoints could be loaded for FoldAvg.')
    print(f"[model] Loaded {len(models)} folds for mean attention.")
    return models


def predict_for_models(
    models: List[torch.nn.Module],
    features: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Average predicted probabilities across the selected fold models."""
    probs = []
    for model in models:
        logits = model(features, mask)
        probs.append(F.softmax(logits, dim=1).cpu().numpy().squeeze())
    return np.mean(np.stack(probs, axis=0), axis=0)


def attention_for_models(
    models: List[torch.nn.Module],
    features: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Compute attention for each selected fold model and average the results."""
    attns = [compute_abmil_attention(m, features, mask) for m in models]
    return np.mean(np.stack(attns, axis=0), axis=0)


# Standalone attention dot maps

def save_attention_dotmap(
    out_path_png: Path,
    attn_vis: np.ndarray,
    coords: np.ndarray,
    cmap: str,
    title: str,
    tile_size: int,
    fill_ratio: float = 0.82,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
) -> None:
    """Draw one dot per tile on a regular grid with colors mapped to attention."""
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        print(f"[skip] Empty coordinates for dot map: {out_path_png.name}")
        return

    step = max(int(tile_size), 1)
    cols = ((coords[:, 0] - coords[:, 0].min()) // step).astype(np.int64)
    rows = ((coords[:, 1] - coords[:, 1].min()) // step).astype(np.int64)
    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1

    valid = attn_vis[np.isfinite(attn_vis)]
    if valid.size == 0:
        print(f"[skip] No finite attention values: {out_path_png.name}")
        return
    vmin = float(np.percentile(valid, clip_lo))
    vmax = float(np.percentile(valid, clip_hi))
    if vmax <= vmin:
        vmin, vmax = float(valid.min()), float(valid.max())
        if vmax <= vmin:
            vmax = vmin + 1e-12

    base = 9.0
    if n_cols >= n_rows:
        fig_w = base
        fig_h = max(3.0, base * (n_rows / max(n_cols, 1)))
    else:
        fig_h = base
        fig_w = max(3.0, base * (n_cols / max(n_rows, 1)))
    fig_w += 1.0
    fig_h += 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_axis_off()

    cell_inch = min(fig_w / max(n_cols, 1), fig_h / max(n_rows, 1)) * 0.92
    cell_pt = cell_inch * 72.0
    marker_size = max(4.0, (cell_pt * float(fill_ratio)) ** 2)

    sc = ax.scatter(
        cols.astype(np.float32),
        rows.astype(np.float32),
        s=marker_size,
        c=attn_vis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker="o",
        linewidths=0,
    )
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Attention")

    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[saved] {out_path_png}")


def build_attention_dot_overlay(
    slide_path: str,
    attn_vis: np.ndarray,
    coords: np.ndarray,
    cmap: str,
    tile_size: int,
    downsample: int,
    alpha: float = 0.75,
    fill_ratio: float = 0.82,
    darken_bg: float = 0.8,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
) -> OverlayLayers:
    """Draw discrete attention dots at level-0 tile coordinates over the WSI thumbnail."""
    coords = np.asarray(coords, dtype=np.float64)
    attn_vis = np.asarray(attn_vis, dtype=np.float64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        raise ValueError('Cannot build a WSI dot overlay from empty or invalid coordinates.')
    if attn_vis.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Attention and coordinate counts differ: attn={attn_vis.shape[0]}, coords={coords.shape[0]}"
        )

    thumbnail = load_wsi_thumbnail(slide_path, downsample)
    overlay = thumbnail.copy()
    if darken_bg < 1.0:
        overlay *= float(darken_bg)

    valid = attn_vis[np.isfinite(attn_vis)]
    if valid.size == 0:
        return layers_from_dot_overlay(
            thumbnail, overlay, overlay, np.zeros(overlay.shape[:2], dtype=bool),
            darken_bg=darken_bg,
        )
    vmin = float(np.percentile(valid, clip_lo))
    vmax = float(np.percentile(valid, clip_hi))
    if vmax <= vmin:
        vmin, vmax = float(valid.min()), float(valid.max())
        if vmax <= vmin:
            vmax = vmin + 1e-12

    norm = np.clip((attn_vis - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    colors = plt.get_cmap(cmap)(norm)[..., :3].astype(np.float32)

    centers_x = np.round((coords[:, 0] + tile_size / 2.0) / downsample).astype(np.int64)
    centers_y = np.round((coords[:, 1] + tile_size / 2.0) / downsample).astype(np.int64)
    radius = max(1, int(round((tile_size / max(downsample, 1)) * float(fill_ratio) / 2.0)))

    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    circle_mask = (xx * xx + yy * yy) <= radius * radius
    dot_alpha = float(np.clip(alpha, 0.0, 1.0))
    heat_rgb = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)

    for cx, cy, color in zip(centers_x, centers_y, colors):
        x0, x1 = cx - radius, cx + radius + 1
        y0, y1 = cy - radius, cy + radius + 1
        if x1 <= 0 or y1 <= 0 or x0 >= overlay.shape[1] or y0 >= overlay.shape[0]:
            continue

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(overlay.shape[1], x1)
        dst_y1 = min(overlay.shape[0], y1)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        local_mask = circle_mask[src_y0:src_y1, src_x0:src_x1]
        region = overlay[dst_y0:dst_y1, dst_x0:dst_x1, :]
        region[local_mask] = (1.0 - dot_alpha) * region[local_mask] + dot_alpha * color
        heat_rgb[dst_y0:dst_y1, dst_x0:dst_x1][local_mask] = color
        paint_mask[dst_y0:dst_y1, dst_x0:dst_x1] |= local_mask

    return layers_from_dot_overlay(
        thumbnail, overlay, heat_rgb, paint_mask, darken_bg=darken_bg
    )


def save_attention_dot_overlay(
    out_path_png: Path,
    layers: OverlayLayers,
    out_path_pdf: Optional[Path] = None,
    alpha: float = 1.0,
) -> None:
    """Save the composite PNG and an editable layered PDF."""
    out_path_pdf = out_path_pdf or out_path_png.with_suffix(".pdf")
    save_overlay_outputs(layers, out_path_png, out_path_pdf, alpha=alpha)
    print(f"[saved] {out_path_png}")


# Per-case processing

def find_slide_for_case(
    case_id: str,
    slide: Optional[Path],
    slide_dir: Optional[Path],
    exts: Tuple[str, ...],
) -> Optional[Path]:
    if slide is not None and slide.is_file():
        return slide
    if slide_dir is not None and slide_dir.is_dir():
        for ext in exts:
            cand = slide_dir / f"{case_id}{ext}"
            if cand.exists():
                return cand
        matches = sorted(slide_dir.glob(f"{case_id}*"))
        matches = [m for m in matches if m.suffix.lower() in exts]
        if len(matches) == 1:
            return matches[0]
    return None


def process_one_h5(
    h5_path: Path,
    models: List[torch.nn.Module],
    class_names: List[str],
    out_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    output_subdir: Optional[str] = None,
    model_label: Optional[str] = None,
) -> None:
    case_id = h5_path.stem
    outdir = out_root / case_id
    if output_subdir:
        outdir = outdir / output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_label = model_label or f"fold_{args.fold}"

    print(f"\n{'=' * 80}")
    print(f"[processing] {h5_path.name} | {model_label}")

    # Compute attention over all tiles and defer tumor filtering to plotting.
    features_np, coords_np = load_h5(str(h5_path))
    n_patches, feat_dim = features_np.shape
    print(f"[data] patches={n_patches}  feat_dim={feat_dim}")

    features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, n_patches, dtype=torch.bool, device=device)

    # Average predicted probabilities across the same folds as attention.
    with torch.no_grad():
        probs = predict_for_models(models, features, mask)
        pred_class = int(np.argmax(probs))
    print('[prediction]')
    for i, name in enumerate(class_names):
        marker = " ← predicted" if i == pred_class else ""
        print(f"   {name}: {probs[i]:.4f}{marker}")

    # Compute attention.
    attn = attention_for_models(models, features, mask)   # Raw attention, shape (N,)
    print(f"[attention] min={attn.min():.3e}  max={attn.max():.3e}  "
          f"sum={attn.sum():.4f}  mean={attn.mean():.3e}")

    attn_vis = normalize_attention(attn, args.attn_norm)
    # Save NPZ data for rendering with plot_attention_from_npz.py.

    # Save raw and display-normalized attention.
    npz_path = outdir / "attributions.npz"
    np.savez_compressed(
        npz_path,
        patch_attributions=attn,          # Use the field names shared by the attribution scripts.
        attention=attn,
        attention_vis=attn_vis,
        attn_norm=args.attn_norm,
        coords=coords_np,
        case_id=case_id,
        probs=probs,
        pred_class=pred_class,
        class_names=class_names,
        model_label=model_label,
        n_models=len(models),
    )
    print(f"[saved] {npz_path}")
    print('[output] Use plot_attention_from_npz.py to render the saved NPZ file.')


def parse_args() -> argparse.Namespace:
    """Parse calculation-only options; plotting is handled by plot_attention_from_npz.py."""
    p = argparse.ArgumentParser(description='Compute ABMIL attention and save an NPZ file.')
    p.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG_JSON)
    p.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    p.add_argument("--average-folds", action="store_true", default=DEFAULT_AVERAGE_FOLDS)
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--model-class", type=str, default=DEFAULT_MODEL_CLASS)
    p.add_argument("--model-params", type=str, default=DEFAULT_MODEL_PARAMS)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--class-names", nargs="+", default=DEFAULT_CLASS_NAMES)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--attn-norm", choices=list(ATTN_NORM_MODES), default=DEFAULT_ATTN_NORM)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    models, class_names = load_models(args, device)
    average_models = load_average_fold_models(args, device)
    print(f"[config] Classes: {class_names}")

    h5_paths = iter_h5(args.h5)
    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 files found: {args.h5}")
    print(f"[config] HDF5 files to process: {len(h5_paths)}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    for h5_path in h5_paths:
        try:
            process_one_h5(
                h5_path=h5_path,
                models=models,
                class_names=class_names,
                out_root=args.outdir,
                args=args,
                device=device,
                model_label=f"fold_{args.fold}",
            )
            if average_models is not None:
                process_one_h5(
                    h5_path=h5_path,
                    models=average_models,
                    class_names=class_names,
                    out_root=args.outdir,
                    args=args,
                    device=device,
                    output_subdir="average_folds",
                    model_label=f"average_folds_n{len(average_models)}",
                )
        except Exception as e:
            print(f"[error] {h5_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 80}\n[complete] Output root: {args.outdir}")


if __name__ == "__main__":
    main()
