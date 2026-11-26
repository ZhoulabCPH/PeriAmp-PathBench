"""
Compare tile attention and prediction groups across models.
Reuse batch attention outputs to draw overlap counts, prediction-group contrasts, and evidence status maps. Average and rank-normalize attention within each model, then select a reproducible top-k set. The groups describe agreement with the ensemble reference class and carry no causal interpretation or accuracy judgment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "image.composite_image": False,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    is_color_like,
)
from matplotlib.patches import Patch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import openslide  # noqa: F401
    _HAS_OPENSLIDE = True
except ImportError:
    _HAS_OPENSLIDE = False

from integrated_gradients import (  # noqa: E402
    make_attribution_canvas,
    signed_asinh_norm,
)
from heatmap_common import (  # noqa: E402
    format_seconds,
    sanitize,
    find_slide_for_stem,
)
from overlay_pdf import (  # noqa: E402
    build_scalar_overlay_layers,
    save_overlay_outputs,
)
from tumor_region import (  # noqa: E402
    add_tumor_region_arguments,
    apply_tumor_filter,
    expand_tumor_plot_arrays,
    tumor_filter_kwargs,
)


# Batch attention root containing attentions and summary.csv
DEFAULT_ATTN_ROOT = r"outputs/interpretability/attention/all_attention"

# An empty encoder list uses every model in the summary.
DEFAULT_MODELS: List[str] = []          # Example labels: GigaPath, H-optimus-0, Virchow2, and UNI
DEFAULT_MIL = "ABMIL"
DEFAULT_FOLD = "Avg"                     # Use Avg for FoldAvg or a numeric fold index.
DEFAULT_CASES: List[str] = []           # An empty case list selects all cases.
DEFAULT_MIN_MODELS = 2

DEFAULT_OUTDIR = r"outputs/interpretability/attention/multimodel_map_plot"
DEFAULT_SLIDE_DIR = r"data/slides"

DEFAULT_TOP_P = 0.10                     # Treat each model's top fraction of patches as high attention.
DEFAULT_ATTR_TOP_P = 0.10
DEFAULT_ATTR_MODE = "absolute"

DEFAULT_TILE_SIZE = 256
DEFAULT_DOWNSAMPLE = 16

# WSI overlay settings
DEFAULT_OVERLAY_ENABLED = True
DEFAULT_ALPHA = 0.70
DEFAULT_SMOOTH_SIGMA = 2.0
DEFAULT_DARKEN_BG = 1.0
DEFAULT_OVERLAP_ZERO_COLOR = "#E6E6E6"
DEFAULT_PDF_FONT_FAMILY = "Arial"

DEFAULT_PANEL_ENABLED = True            # Optionally export the three-panel summary.

# Prefer the batch summary CSV when no explicit path is provided.
# Fall back to NPZ filenames. Synthetic testing uses generated inputs.
DEFAULT_SUMMARY_CSV = None
DEFAULT_SYNTHETIC_TEST = False

# Tumor annotation settings
DEFAULT_TUMOR_FILTER = False                    # Use tumor annotations to identify tumor patches when enabled.
DEFAULT_ANNOTATION_DIR = Path(
    r"outputs/interpretability/tumor_area_output"
)
DEFAULT_ANNOTATION_CSV: Optional[Path] = None  # Prefer this annotation file when processing one slide.
DEFAULT_TUMOR_DISPLAY_MODE = "zero_outside"

# ============================================================

CAT_OVERLAP = "attention_overlap_count"
CAT_CONTRAST = "prediction_group_contrast"
CAT_UNIQUE_STATUS = "unique_evidence_status"
CAT_PANEL = "panel"
CAT_OVERLAP_OVERLAY = "attention_overlap_count_overlay"
CAT_CONTRAST_OVERLAY = "prediction_group_contrast_overlay"
CAT_NPZ = "npz"

ATTR_MODES = ("absolute", "positive", "negative")

STATUS_OTHER = 0
STATUS_ALIGNED_UNIQUE_EVIDENCE = 1
STATUS_DISCORDANT_UNIQUE_EVIDENCE = 2
STATUS_ATTENTION_ONLY_UNIQUE = 3

REASON_NONE = ""
REASON_WEAK_ATTRIBUTION = "weak_attribution"
REASON_ATTRIBUTION_MISSING = "attribution_missing"
REASON_PREDICTION_INFO_MISSING = "prediction_info_missing"

# Use the 99th percentile of absolute contrast for the upper color limit by default.
# The lower percentile sets the asinh linear-region width.
CONTRAST_CLIP_LO = 1.0
CONTRAST_CLIP_HI = 99.0


# Path helpers

def _cat_path(outroot: Path, category: str, prefix: str, ext: str) -> Path:
    subdir = outroot / category
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{prefix}-{category}.{ext.lstrip('.')}"


def _cat_path_pdf(outroot: Path, category: str, prefix: str) -> Path:
    return _cat_path(outroot / "pdf", category, prefix, "pdf")


# Index attention outputs by case and model.

def _fold_token(fold: str) -> str:
    """Map the requested fold to the filename label, such as FoldAvg or Fold0."""
    f = str(fold).strip()
    if f.lower() in ("avg", "foldavg", "average"):
        return "Avg"
    return f


def build_index_from_summary(
    summary_csv: Path,
    mil: str,
    fold: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Index successful summary rows for the chosen MIL method and fold by case and encoder. Return the index and encoder order."""
    fold_tok = _fold_token(fold)
    fold_match = "Avg" if fold_tok == "Avg" else fold_tok

    index: Dict[str, Dict[str, str]] = {}
    models_seen: List[str] = []
    with open(summary_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("error") or "").strip():
                continue
            if (row.get("mil") or "").strip() != mil:
                continue
            if str(row.get("fold") or "").strip() != fold_match:
                continue
            case = (row.get("case") or "").strip()
            feat = (row.get("feature_label") or "").strip()
            prefix = (row.get("file_prefix") or "").strip()
            if not case or not feat or not prefix:
                continue
            index.setdefault(case, {})[feat] = prefix
            if feat not in models_seen:
                models_seen.append(feat)
    return index, models_seen


def build_index_from_attention_npz(
    attn_dir: Path,
    mil: str,
    fold: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Build the case and encoder index from attention filenames when the summary CSV is unavailable."""
    fold_tok = _fold_token(fold)
    index: Dict[str, Dict[str, str]] = {}
    models_seen: List[str] = []
    suffix = "-attentions"

    for npz_path in sorted(attn_dir.glob("*-attentions.npz")):
        body = npz_path.stem
        if not body.endswith(suffix):
            continue
        body = body[:-len(suffix)]
        parts = body.split("-")
        fold_idx = None
        fold_value = None
        for i, part in enumerate(parts):
            match = re.fullmatch(r"[Ff]old(Avg|\d+)", part)
            if match:
                fold_idx = i
                fold_value = match.group(1)
                break
        if fold_idx is None or fold_idx < 3:
            continue
        if fold_tok == "Avg":
            if str(fold_value).lower() != "avg":
                continue
        elif str(fold_value) != fold_tok:
            continue

        case = parts[0]
        feature = parts[1]
        mil_name = "-".join(parts[2:fold_idx])
        if not case or not feature or mil_name != mil:
            continue
        index.setdefault(case, {})[feature] = body
        if feature not in models_seen:
            models_seen.append(feature)

    return index, models_seen


# Load and align tile coordinates.

def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _load_model_npz(npz_path: Path) -> dict:
    """Load attention and coordinates for one model. The caller can skip invalid data."""
    with np.load(npz_path, allow_pickle=True) as d:
        if "attention" not in d.files or "coords" not in d.files:
            raise ValueError('Missing attention or coords.')
        record = {
            "attention": np.asarray(d["attention"], dtype=np.float64).reshape(-1),
            "coords": np.asarray(d["coords"], dtype=np.int64),
            "patch_attributions": None,
            "probs": None,
            "pred_class": None,
            "class_names": None,
        }
        if "patch_attributions" in d.files:
            record["patch_attributions"] = np.asarray(
                d["patch_attributions"], dtype=np.float64
            ).reshape(-1)
        if "probs" in d.files:
            record["probs"] = np.asarray(d["probs"], dtype=np.float64).reshape(-1)
        if "pred_class" in d.files:
            raw_pred = np.asarray(d["pred_class"]).reshape(-1)
            if raw_pred.size:
                try:
                    record["pred_class"] = int(raw_pred[0])
                except (TypeError, ValueError):
                    record["pred_class"] = None
        if "class_names" in d.files:
            try:
                record["class_names"] = [
                    _text(x) for x in np.asarray(d["class_names"]).reshape(-1)
                ]
            except Exception:
                record["class_names"] = None
    return record


def _average_rank_normalize(values: np.ndarray) -> np.ndarray:
    """Average ranks within ties and scale finite values to [0, 1]. Leave nonfinite entries as NaN."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(x))
    n = valid_idx.size
    if n == 0:
        return out
    vals = x[valid_idx]
    order = np.argsort(vals, kind="stable")
    sorted_vals = vals[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0  # 1-based ranks
        ranks[order[start:stop]] = average_rank
        start = stop
    out[valid_idx] = (ranks - 1.0) / max(n - 1, 1)
    return out


def _attribution_strength(values: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if mode == "absolute":
        return np.abs(x)
    if mode == "positive":
        return np.maximum(x, 0.0)
    if mode == "negative":
        return np.maximum(-x, 0.0)
    raise ValueError(f"Unknown attribution mode: {mode}")


def assemble_model_data(
    model_npz: Dict[str, Path],
    models: Sequence[str],
    attr_mode: str,
) -> dict:
    """Align model arrays to the union of tile coordinates and use NaN for missing entries."""
    records: List[dict] = []
    used_models: List[str] = []
    for model in models:
        path = model_npz.get(model)
        if path is None or not path.exists():
            print(f"    [skip] Model {model} has no NPZ file: {path}")
            continue
        try:
            rec = _load_model_npz(path)
            coords = rec["coords"]
            attention = rec["attention"]
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"Expected coordinates with shape (N, 2), found {coords.shape}")
            if attention.size != coords.shape[0]:
                raise ValueError(
                    f"Attention and coordinate lengths differ: {attention.size}/{coords.shape[0]}"
                )
            if attention.size == 0 or not np.isfinite(attention).any():
                raise ValueError('Attention has no finite patch values.')
            if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
                raise ValueError('Coordinates contain duplicate positions.')
            records.append(rec)
            used_models.append(model)
        except Exception as exc:
            print(f"    [skip] Model {model} could not be loaded: {exc}")

    if not records:
        raise RuntimeError('No model attention arrays could be loaded.')

    coords_union = np.unique(
        np.concatenate([rec["coords"] for rec in records], axis=0), axis=0
    )
    key_to_row = {
        (int(x), int(y)): i for i, (x, y) in enumerate(coords_union)
    }
    C, M = coords_union.shape[0], len(records)
    attention_raw = np.full((C, M), np.nan, dtype=np.float64)
    attention_rank = np.full((C, M), np.nan, dtype=np.float64)
    attr_raw = np.full((C, M), np.nan, dtype=np.float64)
    attr_strength = np.full((C, M), np.nan, dtype=np.float64)
    attr_valid_model_mask = np.zeros(M, dtype=bool)

    for j, rec in enumerate(records):
        coords = rec["coords"]
        rows = np.fromiter(
            (key_to_row[(int(x), int(y))] for x, y in coords),
            dtype=np.int64,
            count=coords.shape[0],
        )
        raw = rec["attention"]
        attention_raw[rows, j] = raw
        attention_rank[rows, j] = _average_rank_normalize(raw)
        attrs = rec["patch_attributions"]
        if attrs is not None and attrs.size == coords.shape[0]:
            attr_raw[rows, j] = attrs
            attr_strength[rows, j] = _attribution_strength(attrs, attr_mode)
            attr_valid_model_mask[j] = bool(np.isfinite(attrs).any())

    return {
        "coords": coords_union,
        "models": used_models,
        "records": records,
        "attention_raw": attention_raw,
        "attention_rank": attention_rank,
        "patch_attributions_raw": attr_raw,
        "attribution_strength": attr_strength,
        "attribution_valid_model_mask": attr_valid_model_mask,
    }


# Attention agreement metrics

def exact_topk_mask(values: np.ndarray, top_p: float) -> Tuple[np.ndarray, int]:
    """Select exactly ceil(top_p * N) finite values with a stable order for ties."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.zeros(x.shape, dtype=bool)
    valid_idx = np.flatnonzero(np.isfinite(x))
    if valid_idx.size == 0:
        return mask, 0
    k = min(valid_idx.size, int(np.ceil(top_p * valid_idx.size)))
    if k <= 0:
        return mask, 0
    order = np.argsort(-x[valid_idx], kind="stable")
    mask[valid_idx[order[:k]]] = True
    return mask, k


def compute_attention_overlap(attention_rank: np.ndarray, top_p: float) -> dict:
    C, M = attention_rank.shape
    present = np.isfinite(attention_rank)
    present_count = present.sum(axis=1).astype(np.int64)
    high = np.zeros((C, M), dtype=bool)
    top_k = np.zeros(M, dtype=np.int64)
    for j in range(M):
        high[:, j], top_k[j] = exact_topk_mask(attention_rank[:, j], top_p)
    overlap_count = high.sum(axis=1).astype(np.int64)
    overlap_fraction = np.full(C, np.nan, dtype=np.float64)
    valid = present_count > 0
    overlap_fraction[valid] = overlap_count[valid] / present_count[valid]
    unique_mask = valid & (overlap_count == 1)
    unique_model_idx = np.full(C, -1, dtype=np.int64)
    if M:
        owner = np.argmax(high, axis=1)
        unique_model_idx[unique_mask] = owner[unique_mask]
    return {
        "present": present,
        "present_count": present_count,
        "valid_cell": valid,
        "high_attention": high,
        "top_k_per_model": top_k,
        "overlap_count": overlap_count,
        "overlap_fraction": overlap_fraction,
        "unique_mask": unique_mask,
        "unique_model_idx": unique_model_idx,
    }


def align_prediction_data(records: Sequence[dict], models: Sequence[str]) -> dict:
    """Select the most common class set and align compatible probabilities to its order."""
    M = len(records)
    individually_valid: List[Tuple[int, Tuple[str, ...], np.ndarray]] = []
    warnings: List[str] = []
    invalid_reason: Dict[int, str] = {}
    for j, rec in enumerate(records):
        probs = rec.get("probs")
        names = rec.get("class_names")
        if probs is None:
            invalid_reason[j] = "missing_probs"
            continue
        if not names:
            invalid_reason[j] = "missing_class_names"
            continue
        names_tuple = tuple(_text(x) for x in names)
        if len(set(names_tuple)) != len(names_tuple):
            invalid_reason[j] = "duplicate_class_names"
            continue
        if probs.size != len(names_tuple):
            invalid_reason[j] = "probability_length_mismatch"
            continue
        if not np.isfinite(probs).all():
            invalid_reason[j] = "nonfinite_probs"
            continue
        individually_valid.append((j, names_tuple, probs))

    class_names: List[str] = []
    if individually_valid:
        set_counts: Dict[frozenset, int] = {}
        first_order: Dict[frozenset, Tuple[str, ...]] = {}
        for _, names, _ in individually_valid:
            key = frozenset(names)
            set_counts[key] = set_counts.get(key, 0) + 1
            first_order.setdefault(key, names)
        chosen_set = max(set_counts, key=lambda key: set_counts[key])
        class_names = list(first_order[chosen_set])
    else:
        chosen_set = frozenset()

    K = len(class_names)
    aligned_probs = np.full((M, K), np.nan, dtype=np.float64)
    valid_mask = np.zeros(M, dtype=bool)
    pred_classes = np.full(M, -1, dtype=np.int64)
    pred_names = np.full(M, "", dtype=f"<U{max([len(x) for x in class_names] + [1])}")

    for j, names, probs in individually_valid:
        if frozenset(names) != chosen_set:
            invalid_reason[j] = "class_set_mismatch"
            continue
        source_pos = {name: i for i, name in enumerate(names)}
        aligned = np.asarray([probs[source_pos[name]] for name in class_names])
        pred = int(np.argmax(aligned))
        aligned_probs[j] = aligned
        valid_mask[j] = True
        pred_classes[j] = pred
        pred_names[j] = class_names[pred]

        stored = records[j].get("pred_class")
        if stored is None:
            warnings.append(f"{models[j]}: pred_class missing; recomputed from probs")
        elif not (0 <= stored < len(names)):
            warnings.append(f"{models[j]}: pred_class out of range; recomputed from probs")
        elif names[stored] != class_names[pred]:
            warnings.append(f"{models[j]}: pred_class disagrees with aligned probs")

    for j, reason in invalid_reason.items():
        warnings.append(f"{models[j]}: prediction excluded ({reason})")

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size:
        ensemble_probs = np.mean(aligned_probs[valid_mask], axis=0)
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_name = class_names[ensemble_class]
    else:
        ensemble_probs = np.full(K, np.nan, dtype=np.float64)
        ensemble_class = -1
        ensemble_name = ""

    aligned_mask = valid_mask & (pred_classes == ensemble_class)
    discordant_mask = valid_mask & (pred_classes != ensemble_class)
    return {
        "class_names": class_names,
        "aligned_probs": aligned_probs,
        "model_pred_classes": pred_classes,
        "model_pred_class_names": pred_names,
        "ensemble_probs": ensemble_probs,
        "ensemble_class": ensemble_class,
        "ensemble_class_name": ensemble_name,
        "prediction_valid_model_mask": valid_mask,
        "aligned_model_mask": aligned_mask,
        "discordant_model_mask": discordant_mask,
        "prediction_warnings": warnings,
        "missing_probability_models": [
            models[j] for j in range(M) if not valid_mask[j]
        ],
    }


def _nanmean_columns(values: np.ndarray, column_mask: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    if not np.any(column_mask):
        return out
    selected = values[:, column_mask]
    valid_count = np.isfinite(selected).sum(axis=1)
    sums = np.nansum(selected, axis=1)
    has_value = valid_count > 0
    out[has_value] = sums[has_value] / valid_count[has_value]
    return out


def compute_prediction_group_contrast(
    attention_rank: np.ndarray,
    prediction: dict,
) -> dict:
    valid_mask = prediction["prediction_valid_model_mask"]
    aligned_mask = prediction["aligned_model_mask"]
    discordant_mask = prediction["discordant_model_mask"]
    n_valid = int(valid_mask.sum())
    n_aligned = int(aligned_mask.sum())
    n_discordant = int(discordant_mask.sum())
    aligned_mean = _nanmean_columns(attention_rank, aligned_mask)
    discordant_mean = _nanmean_columns(attention_rank, discordant_mask)
    contrast = np.full(attention_rank.shape[0], np.nan, dtype=np.float64)

    if n_valid < 2:
        status = "insufficient_prediction_models"
    elif n_aligned == 0:
        status = "no_aligned_models"
    elif n_discordant == 0:
        status = "all_predictions_aligned"
    else:
        status = "ok"
        both = np.isfinite(aligned_mean) & np.isfinite(discordant_mean)
        contrast[both] = aligned_mean[both] - discordant_mean[both]
    return {
        "n_prediction_valid_models": n_valid,
        "n_aligned_models": n_aligned,
        "n_discordant_models": n_discordant,
        "aligned_mean_attention": aligned_mean,
        "discordant_mean_attention": discordant_mean,
        "prediction_group_contrast": contrast,
        "contrast_status": status,
    }


def compute_high_attribution(
    attribution_strength: np.ndarray,
    attr_valid_model_mask: np.ndarray,
    attr_top_p: float,
) -> Tuple[np.ndarray, np.ndarray]:
    C, M = attribution_strength.shape
    high = np.zeros((C, M), dtype=bool)
    top_k = np.zeros(M, dtype=np.int64)
    for j in range(M):
        if attr_valid_model_mask[j]:
            high[:, j], top_k[j] = exact_topk_mask(
                attribution_strength[:, j], attr_top_p
            )
    return high, top_k


def compute_unique_evidence_status(
    overlap: dict,
    high_attribution: np.ndarray,
    attribution_strength: np.ndarray,
    attr_valid_model_mask: np.ndarray,
    prediction: dict,
) -> dict:
    C = overlap["unique_mask"].size
    unique_status = np.full(C, STATUS_OTHER, dtype=np.int8)
    reasons = np.full(C, REASON_NONE, dtype="<U64")
    aligned = prediction["aligned_model_mask"]
    discordant = prediction["discordant_model_mask"]
    prediction_valid = prediction["prediction_valid_model_mask"]

    for c in np.flatnonzero(overlap["unique_mask"]):
        owner = int(overlap["unique_model_idx"][c])
        if (
            not attr_valid_model_mask[owner]
            or not np.isfinite(attribution_strength[c, owner])
        ):
            unique_status[c] = STATUS_ATTENTION_ONLY_UNIQUE
            reasons[c] = REASON_ATTRIBUTION_MISSING
        elif not high_attribution[c, owner]:
            unique_status[c] = STATUS_ATTENTION_ONLY_UNIQUE
            reasons[c] = REASON_WEAK_ATTRIBUTION
        elif not prediction_valid[owner]:
            unique_status[c] = STATUS_ATTENTION_ONLY_UNIQUE
            reasons[c] = REASON_PREDICTION_INFO_MISSING
        elif aligned[owner]:
            unique_status[c] = STATUS_ALIGNED_UNIQUE_EVIDENCE
        elif discordant[owner]:
            unique_status[c] = STATUS_DISCORDANT_UNIQUE_EVIDENCE
        else:
            unique_status[c] = STATUS_ATTENTION_ONLY_UNIQUE
            reasons[c] = REASON_PREDICTION_INFO_MISSING
    return {
        "unique_status": unique_status,
        "unique_attention_only_reason": reasons,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman correlation using the Pearson correlation of ranks."""
    n = a.size
    if n < 3:
        return np.nan

    ra = _average_rank_normalize(a)
    rb = _average_rank_normalize(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom <= 0:
        return np.nan
    return float((ra * rb).sum() / denom)


def compute_pairwise(
    V_raw: np.ndarray,
    high: np.ndarray,
    present: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return pairwise Spearman and intersection-over-union matrices using tiles present in both models."""
    M = V_raw.shape[1]
    spear = np.full((M, M), np.nan, dtype=np.float64)
    iou = np.full((M, M), np.nan, dtype=np.float64)
    for i in range(M):
        spear[i, i] = 1.0
        iou[i, i] = 1.0
        for j in range(i + 1, M):
            both = present[:, i] & present[:, j]
            if both.sum() >= 3:
                s = _spearman(V_raw[both, i], V_raw[both, j])
                spear[i, j] = spear[j, i] = s
            hi_i = high[:, i] & both
            hi_j = high[:, j] & both
            union = (hi_i | hi_j).sum()
            inter = (hi_i & hi_j).sum()
            iou[i, j] = iou[j, i] = float(inter / union) if union > 0 else np.nan
    return spear, iou


# Map rendering

def _scalar_canvas(values, coords, tile_size, downsample):
    canvas, mask, origin_xy, _ = make_attribution_canvas(
        values=values, coords=coords, tile_size=tile_size, downsample=downsample,
    )
    return canvas, mask, origin_xy


def _overlap_count_cmap(
    n_levels: int,
    cmap: str,
    zero_color: str,
) -> ListedColormap:
    """Create a discrete count palette shared by standalone maps and overlays."""
    base = plt.get_cmap(cmap, max(n_levels, 1))
    colors = [zero_color] + [base(i) for i in range(n_levels)]
    listed = ListedColormap(colors)
    listed.set_bad(color="white")
    return listed


def render_discrete_count_map(
    values: np.ndarray,
    coords: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    n_levels: int,
    title: str,
    cbar_label: str,
    tile_size: int,
    downsample: int,
    cmap: str = "YlOrRd",
    zero_color: str = DEFAULT_OVERLAP_ZERO_COLOR,
):
    """Draw integer overlap counts with a configurable zero color and white empty background."""
    canvas, _, _ = _scalar_canvas(values, coords, tile_size, downsample)
    if not np.isfinite(canvas).any():
        print(f"    [skip] No finite values: {out_png.name}")
        return
    listed = _overlap_count_cmap(n_levels, cmap, zero_color)
    bounds = np.arange(-0.5, n_levels + 1.0, 1.0)
    norm = BoundaryNorm(bounds, listed.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(canvas, cmap=listed, norm=norm, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        ticks=np.arange(0, n_levels + 1))
    cbar.set_label(cbar_label)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    [saved] {out_png}")




def save_overlay(
    slide_path: Path,
    values: np.ndarray,
    coords: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    cmap: str,
    tile_size: int,
    downsample: int,
    alpha: float,
    smooth_sigma: float,
    darken_bg: float,
    clip_lo: float,
    clip_hi: float,
    symmetric: bool = False,
    norm=None,
):
    canvas, mask, origin_xy = _scalar_canvas(values, coords, tile_size, downsample)
    layers = build_scalar_overlay_layers(
        slide_path=slide_path,
        attr_canvas=canvas,
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
        norm=norm,
    )
    save_overlay_outputs(layers, out_png, out_pdf, alpha=alpha)
    print(f"    [saved] {out_png}")


def render_prediction_contrast(
    values: np.ndarray,
    coords: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    tile_size: int,
    downsample: int,
) -> bool:
    """Draw prediction-group contrast with color limits symmetric around zero."""
    canvas, _, _ = _scalar_canvas(values, coords, tile_size, downsample)
    valid = canvas[np.isfinite(canvas)]
    if valid.size == 0:
        return False
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    # Use symmetric asinh scaling to show weaker contrasts.
    norm = signed_asinh_norm(valid, CONTRAST_CLIP_LO, CONTRAST_CLIP_HI)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(canvas, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean attention difference\n(aligned − discordant)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    [saved] {out_png}")
    return True


def render_unique_evidence_status(
    status: np.ndarray,
    coords: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    tile_size: int,
    downsample: int,
):
    colors = [
        (0.88, 0.88, 0.88, 1.0),  # Other tissue
        (0.20, 0.65, 0.32, 1.0),  # Aligned unique evidence
        (0.55, 0.25, 0.72, 1.0),  # Discordant unique evidence
        (0.90, 0.68, 0.16, 1.0),  # Attention-only unique
    ]
    canvas, _, _ = _scalar_canvas(
        status.astype(np.float64), coords, tile_size, downsample
    )
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(canvas, cmap=cmap, norm=norm, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    handles = [
        Patch(facecolor=colors[1], label="Aligned unique evidence"),
        Patch(facecolor=colors[2], label="Discordant unique evidence"),
        Patch(facecolor=colors[3], label="Attention-only unique"),
        Patch(facecolor=colors[0], label="Other tissue"),
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=10, frameon=False,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    [saved] {out_png}")


def render_three_map_panel(
    overlap_count: np.ndarray,
    contrast: np.ndarray,
    unique_status: np.ndarray,
    coords: np.ndarray,
    M: int,
    contrast_status: str,
    out_png: Path,
    out_pdf: Path,
    case: str,
    tile_size: int,
    downsample: int,
    overlap_zero_color: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis("off")

    overlap_canvas, _, _ = _scalar_canvas(
        overlap_count.astype(float), coords, tile_size, downsample
    )
    count_cmap = _overlap_count_cmap(
        M, "YlOrRd", overlap_zero_color
    )
    axes[0].imshow(
        overlap_canvas, cmap=count_cmap,
        norm=BoundaryNorm(np.arange(-0.5, M + 1.0, 1.0), count_cmap.N),
        interpolation="nearest",
    )
    axes[0].set_title("Attention overlap count")

    if contrast_status == "ok" and np.isfinite(contrast).any():
        contrast_canvas, _, _ = _scalar_canvas(
            contrast, coords, tile_size, downsample
        )
        valid = contrast_canvas[np.isfinite(contrast_canvas)]
        contrast_cmap = plt.get_cmap("RdBu_r").copy()
        contrast_cmap.set_bad("white")
        axes[1].imshow(
            contrast_canvas, cmap=contrast_cmap,
            norm=signed_asinh_norm(valid, CONTRAST_CLIP_LO, CONTRAST_CLIP_HI),
        )
        axes[1].set_title("Prediction-group contrast")
    else:
        message = {
            "all_predictions_aligned": (
                "All valid models predicted the ensemble class.\n"
                "No prediction-group contrast available."
            ),
            "insufficient_prediction_models": (
                "Fewer than two models had valid probabilities.\n"
                "No prediction-group contrast available."
            ),
            "no_aligned_models": (
                "No prediction-aligned models.\n"
                "No prediction-group contrast available."
            ),
        }.get(contrast_status, "No prediction-group contrast available.")
        axes[1].text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
        axes[1].set_title("Prediction-group contrast")

    status_canvas, _, _ = _scalar_canvas(
        unique_status.astype(float), coords, tile_size, downsample
    )
    status_cmap = ListedColormap([
        (0.88, 0.88, 0.88, 1.0),
        (0.20, 0.65, 0.32, 1.0),
        (0.55, 0.25, 0.72, 1.0),
        (0.90, 0.68, 0.16, 1.0),
    ])
    status_cmap.set_bad("white")
    axes[2].imshow(
        status_canvas, cmap=status_cmap,
        norm=BoundaryNorm(np.arange(-0.5, 4.5, 1.0), status_cmap.N),
        interpolation="nearest",
    )
    axes[2].set_title("Unique evidence status")
    fig.suptitle(f"Multi-model attention overview — {case}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"    [saved] {out_png}")


# Per-case execution



def _write_case_pairwise_csv(
    path: Path,
    case: str,
    models: Sequence[str],
    spearman: np.ndarray,
    iou: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "model_a", "model_b", "spearman", "topk_iou"])
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                writer.writerow([
                    case,
                    models[i],
                    models[j],
                    "" if not np.isfinite(spearman[i, j]) else f"{spearman[i, j]:.8g}",
                    "" if not np.isfinite(iou[i, j]) else f"{iou[i, j]:.8g}",
                ])


def _write_unique_model_csv(
    path: Path,
    case: str,
    models: Sequence[str],
    prediction: dict,
    overlap: dict,
    unique_status: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "case", "model", "predicted_class", "prediction_group",
            "n_unique_total", "n_aligned_unique_evidence",
            "n_discordant_unique_evidence", "n_attention_only_unique",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for j, model in enumerate(models):
            owned = overlap["unique_model_idx"] == j
            if prediction["aligned_model_mask"][j]:
                group = "prediction-aligned"
            elif prediction["discordant_model_mask"][j]:
                group = "prediction-discordant"
            else:
                group = "unavailable"
            writer.writerow({
                "case": case,
                "model": model,
                "predicted_class": prediction["model_pred_class_names"][j],
                "prediction_group": group,
                "n_unique_total": int(owned.sum()),
                "n_aligned_unique_evidence": int(
                    (owned & (unique_status == STATUS_ALIGNED_UNIQUE_EVIDENCE)).sum()
                ),
                "n_discordant_unique_evidence": int(
                    (owned & (unique_status == STATUS_DISCORDANT_UNIQUE_EVIDENCE)).sum()
                ),
                "n_attention_only_unique": int(
                    (owned & (unique_status == STATUS_ATTENTION_ONLY_UNIQUE)).sum()
                ),
            })


def run_case(
    case: str,
    model_npz: Dict[str, Path],
    models: List[str],
    outroot: Path,
    slide_path: Optional[Path],
    args: argparse.Namespace,
    tag: str,
) -> Tuple[dict, List[str], np.ndarray, np.ndarray]:
    data = assemble_model_data(model_npz, models, args.attr_mode)
    all_coords = np.asarray(data["coords"])
    all_indices = np.arange(all_coords.shape[0], dtype=np.int64)
    tumor_coords, filtered, filter_info = apply_tumor_filter(
        all_coords,
        case,
        arrays={
            "attention_raw": data["attention_raw"],
            "attention_rank": data["attention_rank"],
            "patch_attributions_raw": data["patch_attributions_raw"],
            "attribution_strength": data["attribution_strength"],
            "indices": all_indices,
        },
        **tumor_filter_kwargs(args),
    )
    data["coords"] = tumor_coords
    data["attention_raw"] = filtered["attention_raw"]
    data["attention_rank"] = filtered["attention_rank"]
    data["patch_attributions_raw"] = filtered["patch_attributions_raw"]
    data["attribution_strength"] = filtered["attribution_strength"]
    tumor_indices = np.asarray(filtered["indices"], dtype=np.int64).reshape(-1)
    coords = data["coords"]
    used_models = data["models"]
    attention_raw = data["attention_raw"]
    attention_rank = data["attention_rank"]
    M, C = len(used_models), coords.shape[0]
    if M < args.min_models:
        raise RuntimeError(
            f"Loaded models: {M} < min_models={args.min_models}"
        )
    overlap_cmap = _overlap_count_cmap(
        M, "YlOrRd", args.overlap_zero_color
    )
    overlap_norm = BoundaryNorm(
        np.arange(-0.5, M + 1.0, 1.0), overlap_cmap.N
    )

    overlap = compute_attention_overlap(attention_rank, args.top_p)
    prediction = align_prediction_data(data["records"], used_models)
    contrast = compute_prediction_group_contrast(attention_rank, prediction)
    high_attr, attr_top_k = compute_high_attribution(
        data["attribution_strength"],
        data["attribution_valid_model_mask"],
        args.attr_top_p,
    )
    unique = compute_unique_evidence_status(
        overlap,
        high_attr,
        data["attribution_strength"],
        data["attribution_valid_model_mask"],
        prediction,
    )
    unique_status = unique["unique_status"]
    overlap_plot_values = np.where(
        overlap["valid_cell"], overlap["overlap_count"], np.nan
    )
    unique_status_plot_values = np.where(
        overlap["valid_cell"], unique_status, np.nan
    )
    spearman, iou = compute_pairwise(
        attention_raw, overlap["high_attention"], overlap["present"]
    )

    display_mode = str(
        getattr(args, "tumor_display_mode", DEFAULT_TUMOR_DISPLAY_MODE)
    )
    plot_coords, plot_arrays = expand_tumor_plot_arrays(
        all_coords,
        tumor_coords,
        tumor_indices,
        {
            "overlap": overlap_plot_values,
            "unique_status": unique_status_plot_values,
            "contrast": contrast["prediction_group_contrast"],
        },
        enabled=bool(filter_info["enabled"]),
        display_mode=display_mode,
    )
    overlap_plot_values = plot_arrays["overlap"]
    unique_status_plot_values = plot_arrays["unique_status"]
    contrast_plot_values = plot_arrays["contrast"]
    overlay_clip_lo = 0.0 if display_mode == "zero_outside" and filter_info["enabled"] else 0.5
    overlay_clip_hi = 100.0 if display_mode == "zero_outside" and filter_info["enabled"] else 99.5
    if filter_info["enabled"]:
        if display_mode == "zero_outside":
            print(
                f"[region] Tumor annotation with zero outside: "
                f"tumor={tumor_coords.shape[0]} / total={plot_coords.shape[0]}"
            )
        else:
            print(
                f"[region] Tumor patches only: "
                f"patches={tumor_coords.shape[0]}"
            )

    prefix = f"{sanitize(case)}-{tag}"
    pct = int(round(args.top_p * 100))
    attr_pct = int(round(args.attr_top_p * 100))
    n_attr_valid = int(data["attribution_valid_model_mask"].sum())
    print(
        f"[Models] total={M}, prediction-valid="
        f"{contrast['n_prediction_valid_models']}, attribution-valid={n_attr_valid}"
    )
    print(f"[Top-k] attention={pct}%, attribution={attr_pct}%")
    if prediction["ensemble_class"] >= 0:
        print(
            f"[Prediction] ensemble={prediction['ensemble_class_name']}, "
            f"aligned={contrast['n_aligned_models']}, "
            f"discordant={contrast['n_discordant_models']}"
        )
    else:
        print("[Prediction] ensemble=unavailable, aligned=0, discordant=0")

    render_discrete_count_map(
        overlap_plot_values,
        plot_coords,
        _cat_path(outroot, CAT_OVERLAP, prefix, "png"),
        _cat_path_pdf(outroot, CAT_OVERLAP, prefix),
        n_levels=M,
        title=f"High-attention overlap — {case} (top {pct}%, M={M})",
        cbar_label="Number of models selecting patch as top-k",
        tile_size=args.tile_size,
        downsample=args.downsample,
        zero_color=args.overlap_zero_color,
    )

    if contrast["contrast_status"] == "ok":
        rendered = render_prediction_contrast(
            contrast_plot_values,
            plot_coords,
            _cat_path(outroot, CAT_CONTRAST, prefix, "png"),
            _cat_path_pdf(outroot, CAT_CONTRAST, prefix),
            title=(
                f"Prediction-group attention contrast — {case}\n"
                f"Ensemble class: {prediction['ensemble_class_name']}; "
                f"aligned={contrast['n_aligned_models']}, "
                f"discordant={contrast['n_discordant_models']}"
            ),
            tile_size=args.tile_size,
            downsample=args.downsample,
        )
        if not rendered:
            contrast["contrast_status"] = "no_patch_with_both_groups"
            print(
                "Skipped prediction_group_contrast: no patch has valid "
                "attention in both prediction groups."
            )
    else:
        skip_message = {
            "all_predictions_aligned": (
                "all valid models predict the ensemble class."
            ),
            "no_aligned_models": "no prediction-aligned models.",
            "insufficient_prediction_models": (
                "fewer than two models have valid probabilities."
            ),
        }.get(contrast["contrast_status"], contrast["contrast_status"])
        print(f"Skipped prediction_group_contrast: {skip_message}")

    render_unique_evidence_status(
        unique_status_plot_values,
        plot_coords,
        _cat_path(outroot, CAT_UNIQUE_STATUS, prefix, "png"),
        _cat_path_pdf(outroot, CAT_UNIQUE_STATUS, prefix),
        title=(
            f"Unique attention evidence status — {case}\n"
            f"Attention top {pct}%; attribution top {attr_pct}%"
        ),
        tile_size=args.tile_size,
        downsample=args.downsample,
    )

    if args.panel_enabled:
        render_three_map_panel(
            overlap_plot_values,
            contrast_plot_values,
            unique_status_plot_values,
            plot_coords,
            M,
            contrast["contrast_status"],
            _cat_path(outroot, CAT_PANEL, prefix, "png"),
            _cat_path_pdf(outroot, CAT_PANEL, prefix),
            case,
            args.tile_size,
            args.downsample,
            args.overlap_zero_color,
        )

    if slide_path is not None and _HAS_OPENSLIDE and args.overlay_enabled:
        try:
            save_overlay(
                slide_path,
                overlap_plot_values,
                plot_coords,
                _cat_path(outroot, CAT_OVERLAP_OVERLAY, prefix, "png"),
                _cat_path_pdf(outroot, CAT_OVERLAP_OVERLAY, prefix),
                cmap=overlap_cmap,
                tile_size=args.tile_size,
                downsample=args.downsample,
                alpha=args.alpha,
                smooth_sigma=0.0,
                darken_bg=args.darken_bg,
                clip_lo=0.0,
                clip_hi=100.0,
                norm=overlap_norm,
            )
            if contrast["contrast_status"] == "ok":
                save_overlay(
                    slide_path,
                    contrast_plot_values,
                    plot_coords,
                    _cat_path(outroot, CAT_CONTRAST_OVERLAY, prefix, "png"),
                    _cat_path_pdf(outroot, CAT_CONTRAST_OVERLAY, prefix),
                    cmap="RdBu_r",
                    tile_size=args.tile_size,
                    downsample=args.downsample,
                    alpha=args.alpha,
                    smooth_sigma=args.smooth_sigma,
                    darken_bg=args.darken_bg,
                    clip_lo=overlay_clip_lo,
                    clip_hi=overlay_clip_hi,
                    symmetric=True,
                )
        except Exception as exc:
            print(f"    [warning] WSI overlay failed: {exc}")

    unique_names = np.full(C, "", dtype=f"<U{max([len(x) for x in used_models] + [1])}")
    for c in np.flatnonzero(overlap["unique_mask"]):
        unique_names[c] = used_models[overlap["unique_model_idx"][c]]
    prediction_valid_models = [
        used_models[j]
        for j in np.flatnonzero(prediction["prediction_valid_model_mask"])
    ]
    npz_path = _cat_path(outroot, CAT_NPZ, prefix, "npz")
    np.savez_compressed(
        npz_path,
        coords=coords,
        models=np.asarray(used_models, dtype=str),
        attention_raw=attention_raw,
        attention_rank=attention_rank,
        present=overlap["present"],
        present_count=overlap["present_count"],
        high_attention=overlap["high_attention"],
        top_p=np.float64(args.top_p),
        top_k_per_model=overlap["top_k_per_model"],
        overlap_count=overlap["overlap_count"],
        overlap_fraction=overlap["overlap_fraction"],
        class_names=np.asarray(prediction["class_names"], dtype=str),
        aligned_probs=prediction["aligned_probs"],
        model_pred_classes=prediction["model_pred_classes"],
        model_pred_class_names=prediction["model_pred_class_names"],
        ensemble_probs=prediction["ensemble_probs"],
        ensemble_class=np.int64(prediction["ensemble_class"]),
        ensemble_class_name=np.asarray(prediction["ensemble_class_name"]),
        prediction_valid_models=np.asarray(prediction_valid_models, dtype=str),
        prediction_valid_model_mask=prediction["prediction_valid_model_mask"],
        aligned_model_mask=prediction["aligned_model_mask"],
        discordant_model_mask=prediction["discordant_model_mask"],
        n_aligned_models=np.int64(contrast["n_aligned_models"]),
        n_discordant_models=np.int64(contrast["n_discordant_models"]),
        aligned_mean_attention=contrast["aligned_mean_attention"],
        discordant_mean_attention=contrast["discordant_mean_attention"],
        prediction_group_contrast=contrast["prediction_group_contrast"],
        contrast_status=np.asarray(contrast["contrast_status"]),
        prediction_warnings=np.asarray(
            prediction["prediction_warnings"], dtype=str
        ),
        patch_attributions_raw=data["patch_attributions_raw"],
        attribution_strength=data["attribution_strength"],
        high_attribution=high_attr,
        attr_top_p=np.float64(args.attr_top_p),
        attr_top_k_per_model=attr_top_k,
        attr_mode=np.asarray(args.attr_mode),
        attribution_valid_model_mask=data["attribution_valid_model_mask"],
        unique_mask=overlap["unique_mask"],
        unique_model_idx=overlap["unique_model_idx"],
        unique_model_name=unique_names,
        unique_status=unique_status,
        unique_attention_only_reason=unique["unique_attention_only_reason"],
        spearman=spearman,
        iou=iou,
    )
    cohort_dir = outroot / "cohort"
    _write_case_pairwise_csv(
        cohort_dir / f"{prefix}-pairwise.csv",
        case,
        used_models,
        spearman,
        iou,
    )
    _write_unique_model_csv(
        cohort_dir / f"{prefix}-unique_evidence_by_model.csv",
        case,
        used_models,
        prediction,
        overlap,
        unique_status,
    )

    valid = overlap["valid_cell"]
    n_valid = int(valid.sum())
    denom = max(n_valid, 1)
    n_unique = int(overlap["unique_mask"].sum())
    n_aligned_unique = int(
        (unique_status == STATUS_ALIGNED_UNIQUE_EVIDENCE).sum()
    )
    n_discordant_unique = int(
        (unique_status == STATUS_DISCORDANT_UNIQUE_EVIDENCE).sum()
    )
    n_attention_only = int(
        (unique_status == STATUS_ATTENTION_ONLY_UNIQUE).sum()
    )
    n_overlap_all = int(
        (valid & (overlap["overlap_count"] == M)).sum()
    )
    ensemble_max = (
        float(np.nanmax(prediction["ensemble_probs"]))
        if prediction["ensemble_probs"].size
        and np.isfinite(prediction["ensemble_probs"]).any()
        else np.nan
    )
    missing_attr_models = [
        used_models[j]
        for j in range(M)
        if not data["attribution_valid_model_mask"][j]
    ]
    result = {
        "case": case,
        "tag": tag,
        "n_models": M,
        "models": ";".join(used_models),
        "n_patches_union": C,
        "top_p": args.top_p,
        "attr_top_p": args.attr_top_p,
        "n_prediction_valid_models": contrast["n_prediction_valid_models"],
        "ensemble_class": prediction["ensemble_class"],
        "ensemble_class_name": prediction["ensemble_class_name"],
        "ensemble_max_probability": ensemble_max,
        "n_aligned_models": contrast["n_aligned_models"],
        "n_discordant_models": contrast["n_discordant_models"],
        "contrast_status": contrast["contrast_status"],
        "n_overlap_zero": int((valid & (overlap["overlap_count"] == 0)).sum()),
        "n_overlap_one": n_unique,
        "n_overlap_all": n_overlap_all,
        "frac_overlap_one": n_unique / denom,
        "frac_overlap_all": n_overlap_all / denom,
        "mean_overlap_count": (
            float(np.mean(overlap["overlap_count"][valid])) if n_valid else np.nan
        ),
        "mean_overlap_fraction": (
            float(np.mean(overlap["overlap_fraction"][valid])) if n_valid else np.nan
        ),
        "n_unique_total": n_unique,
        "n_aligned_unique_evidence": n_aligned_unique,
        "n_discordant_unique_evidence": n_discordant_unique,
        "n_attention_only_unique": n_attention_only,
        "frac_unique_total": n_unique / denom,
        "frac_aligned_unique_evidence": n_aligned_unique / denom,
        "frac_discordant_unique_evidence": n_discordant_unique / denom,
        "frac_attention_only_unique": n_attention_only / denom,
        "missing_probability_models": ";".join(
            prediction["missing_probability_models"]
        ),
        "missing_attribution_models": ";".join(missing_attr_models),
        "prediction_warnings": "; ".join(prediction["prediction_warnings"]),
        "slide": str(slide_path) if slide_path else "",
        "error": "",
    }
    print(
        f"[Overlap] unique={n_unique}, all-model={n_overlap_all}\n"
        f"[Unique evidence] aligned={n_aligned_unique}, "
        f"discordant={n_discordant_unique}, attention-only={n_attention_only}"
    )
    plt.close("all")
    gc.collect()
    return result, used_models, spearman, iou


# ─────────────────── main ───────────────────



SUMMARY_FIELDS = [
    "case", "tag", "n_models", "models", "n_patches_union",
    "top_p", "attr_top_p",
    "n_prediction_valid_models", "ensemble_class", "ensemble_class_name",
    "ensemble_max_probability", "n_aligned_models", "n_discordant_models",
    "contrast_status",
    "n_overlap_zero", "n_overlap_one", "n_overlap_all",
    "frac_overlap_one", "frac_overlap_all",
    "mean_overlap_count", "mean_overlap_fraction",
    "n_unique_total", "n_aligned_unique_evidence",
    "n_discordant_unique_evidence", "n_attention_only_unique",
    "frac_unique_total", "frac_aligned_unique_evidence",
    "frac_discordant_unique_evidence", "frac_attention_only_unique",
    "missing_probability_models", "missing_attribution_models",
    "prediction_warnings", "slide", "error",
]


def _write_summary(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_cohort_pairwise(
    cohort_dir: Path,
    tag: str,
    models: Sequence[str],
    mean_spearman: np.ndarray,
    mean_iou: np.ndarray,
    spearman_count: np.ndarray,
    iou_count: np.ndarray,
):
    cohort_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cohort_dir / f"cohort_{tag}_pairwise.npz",
        models=np.asarray(models, dtype=str),
        mean_spearman=mean_spearman,
        mean_iou=mean_iou,
        spearman_case_count=spearman_count,
        iou_case_count=iou_count,
    )
    for name, matrix in (
        ("spearman", mean_spearman),
        ("iou", mean_iou),
    ):
        with open(
            cohort_dir / f"cohort_{tag}_{name}.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + list(models))
            for i, model in enumerate(models):
                writer.writerow([
                    model,
                    *[
                        "" if not np.isfinite(matrix[i, j])
                        else f"{matrix[i, j]:.8g}"
                        for j in range(len(models))
                    ],
                ])


def run_synthetic_test(args: argparse.Namespace):
    """Use synthetic inputs to verify the three evidence states and file outputs."""
    outroot = Path(args.outdir) / "synthetic_test"
    coords = np.asarray(
        [[0, 0], [256, 0], [512, 0], [768, 0], [1024, 0], [1280, 0]],
        dtype=np.int64,
    )
    models = ["ModelA", "ModelB", "ModelC"]
    synthetic = {
        "ModelA": {
            "attention": [10, 9, 1, 1, 0, 0],
            "patch_attributions": [1, 10, 0, 0, 0, 9],
            "probs": [0.9, 0.1],
            "class_names": ["ClassA", "ClassB"],
            "pred_class": 0,
        },
        "ModelB": {
            "attention": [10, 1, 9, 0, 0, 0],
            "patch_attributions": [9, 0, 1, 0, 0, 10],
            "probs": [0.2, 0.8],
            "class_names": ["ClassB", "ClassA"],
            "pred_class": 1,
        },
        "ModelC": {
            "attention": [0, 0, 0, 10, 9, 1],
            "patch_attributions": [0, 0, 0, 10, 1, 9],
            "probs": [0.2, 0.8],
            "class_names": ["ClassA", "ClassB"],
            "pred_class": 1,
        },
    }
    with tempfile.TemporaryDirectory(prefix="attention_consensus_synthetic_") as tmp:
        tmpdir = Path(tmp)
        model_npz: Dict[str, Path] = {}
        for model, values in synthetic.items():
            path = tmpdir / f"{model}.npz"
            np.savez_compressed(
                path,
                coords=coords,
                attention=np.asarray(values["attention"], dtype=np.float64),
                patch_attributions=np.asarray(
                    values["patch_attributions"], dtype=np.float64
                ),
                probs=np.asarray(values["probs"], dtype=np.float64),
                class_names=np.asarray(values["class_names"], dtype=str),
                pred_class=np.int64(values["pred_class"]),
            )
            model_npz[model] = path
        test_args = argparse.Namespace(**vars(args))
        test_args.top_p = 0.20
        test_args.attr_top_p = 0.20
        test_args.attr_mode = "absolute"
        test_args.min_models = 2
        test_args.overlay_enabled = False
        test_args.tumor_filter = False
        result, _, _, _ = run_case(
            case="synthetic_case",
            model_npz=model_npz,
            models=models,
            outroot=outroot,
            slide_path=None,
            args=test_args,
            tag="Synthetic",
        )

    npz_path = (
        outroot / CAT_NPZ / "synthetic_case-Synthetic-npz.npz"
    )
    with np.load(npz_path, allow_pickle=False) as saved:
        assert saved["top_k_per_model"].tolist() == [2, 2, 2]
        assert saved["attr_top_k_per_model"].tolist() == [2, 2, 2]
        assert saved["attention_rank"][2, 0] == saved["attention_rank"][3, 0]
        assert saved["unique_status"].tolist() == [
            STATUS_OTHER,
            STATUS_ALIGNED_UNIQUE_EVIDENCE,
            STATUS_ATTENTION_ONLY_UNIQUE,
            STATUS_DISCORDANT_UNIQUE_EVIDENCE,
            STATUS_ATTENTION_ONLY_UNIQUE,
            STATUS_OTHER,
        ]
        assert saved["unique_attention_only_reason"][2] == REASON_WEAK_ATTRIBUTION
        assert saved["ensemble_class_name"].item() == "ClassA"
        assert saved["contrast_status"].item() == "ok"

    tied_ranks = _average_rank_normalize(np.asarray([1.0, 1.0, 3.0]))
    assert np.allclose(tied_ranks, [0.25, 0.25, 1.0])
    tied_topk, tied_k = exact_topk_mask(
        np.asarray([5.0, 5.0, 5.0, 1.0]), 0.5
    )
    assert tied_k == 2 and tied_topk.tolist() == [True, True, False, False]

    all_aligned_prediction = align_prediction_data(
        [
            {
                "probs": np.asarray([0.8, 0.2]),
                "class_names": ["ClassA", "ClassB"],
                "pred_class": 0,
            },
            {
                "probs": np.asarray([0.7, 0.3]),
                "class_names": ["ClassA", "ClassB"],
                "pred_class": 0,
            },
        ],
        ["A", "B"],
    )
    all_aligned_contrast = compute_prediction_group_contrast(
        np.asarray([[0.1, 0.2], [0.9, 0.8]]),
        all_aligned_prediction,
    )
    assert all_aligned_contrast["contrast_status"] == "all_predictions_aligned"
    assert not np.isfinite(
        all_aligned_contrast["prediction_group_contrast"]
    ).any()

    expected = [
        outroot / CAT_OVERLAP
        / "synthetic_case-Synthetic-attention_overlap_count.png",
        outroot / CAT_CONTRAST
        / "synthetic_case-Synthetic-prediction_group_contrast.png",
        outroot / CAT_UNIQUE_STATUS
        / "synthetic_case-Synthetic-unique_evidence_status.png",
        outroot / "pdf" / CAT_OVERLAP
        / "synthetic_case-Synthetic-attention_overlap_count.pdf",
        outroot / "pdf" / CAT_CONTRAST
        / "synthetic_case-Synthetic-prediction_group_contrast.pdf",
        outroot / "pdf" / CAT_UNIQUE_STATUS
        / "synthetic_case-Synthetic-unique_evidence_status.pdf",
        npz_path,
    ]
    missing = [str(path) for path in expected if not path.exists()]
    assert not missing, f"Synthetic test output is missing: {missing}"
    print(f"[Synthetic test] PASS: {outroot}")
    print(
        "[Synthetic test] states: aligned=1, discordant=1, "
        "attention-only=2"
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compare attention overlap, prediction-group contrast, and model-specific evidence.'
    )
    parser.add_argument("--attn-root", type=str, default=DEFAULT_ATTN_ROOT)
    parser.add_argument(
        "--summary-csv", type=str, default=DEFAULT_SUMMARY_CSV,
        help='Input index. Defaults to the batch summary CSV, then to the attention NPZ filenames.',
    )
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--mil", type=str, default=DEFAULT_MIL)
    parser.add_argument(
        "--fold", type=str, default=DEFAULT_FOLD,
        help='Avg, FoldAvg, or an individual fold number',
    )
    parser.add_argument("--cases", type=str, nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--min-models", type=int, default=DEFAULT_MIN_MODELS)
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    parser.add_argument("--slide-dir", type=str, default=DEFAULT_SLIDE_DIR)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--attr-top-p", type=float, default=DEFAULT_ATTR_TOP_P)
    parser.add_argument(
        "--attr-mode", choices=list(ATTR_MODES), default=DEFAULT_ATTR_MODE
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument(
        "--overlay", dest="overlay_enabled", action="store_true",
        default=DEFAULT_OVERLAY_ENABLED,
        help='Generate slide overlays for attention overlap and available prediction contrasts',
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--darken-bg", type=float, default=DEFAULT_DARKEN_BG)
    parser.add_argument(
        "--overlap-zero-color",
        type=str,
        default=DEFAULT_OVERLAP_ZERO_COLOR,
        help=(
            'Matplotlib color for zero overlap, for example '
            '#E6E6E6, gray, or none. Applies to both the standalone map and overlay.'
        ),
    )
    parser.add_argument(
        "--pdf-font-family",
        type=str,
        default=DEFAULT_PDF_FONT_FAMILY,
        help='Editable TrueType font for PDF output. Defaults to Arial.',
    )
    parser.add_argument(
        "--panel", dest="panel_enabled", action="store_true",
        default=DEFAULT_PANEL_ENABLED,
        help='Add a three-panel overview',
    )
    parser.add_argument(
        "--synthetic-test", action="store_true", default=DEFAULT_SYNTHETIC_TEST,
        help='Generate a minimal synthetic case and verify the outputs',
    )
    add_tumor_region_arguments(
        parser,
        single_csv=True,
        annotation_dir_default=DEFAULT_ANNOTATION_DIR,
        annotation_csv_default=DEFAULT_ANNOTATION_CSV,
        tumor_filter_default=DEFAULT_TUMOR_FILTER,
        display_mode_default=DEFAULT_TUMOR_DISPLAY_MODE,
    )
    parser.set_defaults(tumor_filter=DEFAULT_TUMOR_FILTER)
    return parser


def main():
    args = build_argument_parser().parse_args()
    matplotlib.rcParams["font.family"] = args.pdf_font_family
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError('--top-p must be in (0, 1].')
    if not (0.0 < args.attr_top_p <= 1.0):
        raise ValueError('--attr-top-p must be in (0, 1].')
    if args.min_models < 2:
        raise ValueError('--min-models must be at least 2.')
    if args.tile_size <= 0 or args.downsample <= 0:
        raise ValueError('--tile-size and --downsample must be positive integers.')
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError('--alpha must be in [0, 1].')
    if not (0.0 <= args.darken_bg <= 1.0):
        raise ValueError('--darken-bg must be in [0, 1].')
    if not is_color_like(args.overlap_zero_color):
        raise ValueError(
            '--overlap-zero-color is not a valid Matplotlib color: '
            f"{args.overlap_zero_color!r}"
        )
    if args.synthetic_test:
        run_synthetic_test(args)
        return

    attn_root = Path(args.attn_root)
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv
        else attn_root / "summary.csv"
    )
    attn_dir = attn_root / "attentions"
    if not attn_dir.is_dir():
        raise NotADirectoryError(f"Attention directory not found: {attn_dir}")

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)
    slide_dir: Optional[Path] = None
    if (
        args.overlay_enabled
        and args.slide_dir
        and str(args.slide_dir).strip().lower() not in ("none", "")
    ):
        slide_dir = Path(args.slide_dir)
        if not slide_dir.is_dir():
            print(f"[warning] Invalid slide directory. Skipping overlays: {slide_dir}")
            slide_dir = None
        elif not _HAS_OPENSLIDE:
            print('[warning] Install openslide-python to generate slide overlays.')
            slide_dir = None

    fold_tok = _fold_token(args.fold)
    tag = f"{args.mil}-Fold{fold_tok}"
    if args.summary_csv:
        if not summary_csv.exists():
            raise FileNotFoundError(f"summary.csv not found: {summary_csv}")
        index, models_seen = build_index_from_summary(
            summary_csv, args.mil, args.fold
        )
        print(f"[index] Using summary.csv: {summary_csv}")
    elif summary_csv.exists():
        index, models_seen = build_index_from_summary(
            summary_csv, args.mil, args.fold
        )
        print(f"[index] Using summary.csv: {summary_csv}")
    else:
        index, models_seen = build_index_from_attention_npz(
            attn_dir, args.mil, args.fold
        )
        print(
            f"[index] No summary CSV found. Using attention NPZ filenames: {attn_dir}"
        )
    if not index:
        raise RuntimeError(
            f"No matching summary rows for mil={args.mil}, fold={args.fold}"
        )
    models = list(args.models) if args.models else models_seen
    if len(models) < 2:
        raise ValueError(
            f"At least two candidate models are required. Found {len(models)}. Use --models to select them."
        )
    print(
        f"[config] mil={args.mil}, fold={args.fold}, top_p={args.top_p}, "
        f"attr_top_p={args.attr_top_p}, attr_mode={args.attr_mode}"
    )
    print(
        f"[tumor annotation] {'enabled at ' + str(args.annotation_dir) if args.tumor_filter else 'disabled'}"
        + (
            f", display={args.tumor_display_mode}"
            if args.tumor_filter else ""
        )
    )
    print(f"[models] {len(models)} selected: {', '.join(models)}")

    cases = args.cases if args.cases else sorted(index)
    plan: List[Tuple[str, Dict[str, Path]]] = []
    for case in cases:
        if case not in index:
            print(f"[skip] case={case} is absent from the attention index.")
            continue
        paths: Dict[str, Path] = {}
        for model in models:
            file_prefix = index[case].get(model)
            if file_prefix:
                path = attn_dir / f"{file_prefix}-attentions.npz"
                if path.exists():
                    paths[model] = path
        if len(paths) < args.min_models:
            print(
                f"[skip] case={case} has available models: {len(paths)} "
                f"< min_models={args.min_models}"
            )
            continue
        plan.append((case, paths))
    if not plan:
        raise RuntimeError('No runnable cases. Check the model count and attention inputs.')

    summary: List[dict] = []
    n_models = len(models)
    model_pos = {model: i for i, model in enumerate(models)}
    spear_sum = np.zeros((n_models, n_models), dtype=np.float64)
    spear_count = np.zeros((n_models, n_models), dtype=np.int64)
    iou_sum = np.zeros((n_models, n_models), dtype=np.float64)
    iou_count = np.zeros((n_models, n_models), dtype=np.int64)
    start = time.time()
    for k, (case, model_npz) in enumerate(plan, 1):
        elapsed = time.time() - start
        rate = (k - 1) / elapsed if elapsed > 0 and k > 1 else 0.0
        eta = (len(plan) - k + 1) / rate if rate > 0 else 0.0
        print(
            f"\n[progress] {k}/{len(plan)} "
            f"({100.0 * k / len(plan):.1f}%), "
            f"elapsed={format_seconds(elapsed)}, eta={format_seconds(eta)}"
        )
        print(f"[Case] {case}")
        slide_path = find_slide_for_stem(slide_dir, case) if slide_dir else None
        try:
            result, used_models, spearman, iou = run_case(
                case, model_npz, models, outroot, slide_path, args, tag
            )
            summary.append(result)
            for a, model_a in enumerate(used_models):
                ia = model_pos[model_a]
                for b, model_b in enumerate(used_models):
                    ib = model_pos[model_b]
                    if np.isfinite(spearman[a, b]):
                        spear_sum[ia, ib] += spearman[a, b]
                        spear_count[ia, ib] += 1
                    if np.isfinite(iou[a, b]):
                        iou_sum[ia, ib] += iou[a, b]
                        iou_count[ia, ib] += 1
        except Exception as exc:
            print(f"[error] case={case}: {exc}")
            traceback.print_exc()
            summary.append({
                "case": case, "tag": tag, "error": str(exc)
            })
        finally:
            plt.close("all")
            gc.collect()

    mean_spearman = np.full_like(spear_sum, np.nan)
    mean_iou = np.full_like(iou_sum, np.nan)
    np.divide(
        spear_sum, spear_count, out=mean_spearman, where=spear_count > 0
    )
    np.divide(iou_sum, iou_count, out=mean_iou, where=iou_count > 0)
    _write_cohort_pairwise(
        outroot / "cohort", tag, models,
        mean_spearman, mean_iou, spear_count, iou_count,
    )
    summary_path = outroot / "summary.csv"
    _write_summary(summary_path, summary)
    # Retain the legacy filenames for downstream readers.
    _write_summary(outroot / "multimodel_summary.csv", summary)
    ok = sum(1 for row in summary if not row.get("error"))
    print(f"\n[summary] {summary_path}")
    print(f"[complete] {len(summary)} cases, succeeded: {ok}, failed: {len(summary) - ok}")
    print(f"[elapsed] {format_seconds(time.time() - start)}")
    print(f"[output] {outroot}")


if __name__ == "__main__":
    main()
