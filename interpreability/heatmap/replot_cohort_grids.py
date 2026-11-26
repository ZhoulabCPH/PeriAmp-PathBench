"""
Rebuild statistics, maps, and patch grids from previously saved cohort clusters.
Load coordinates, case IDs, and cluster assignments from the cluster-result NPZ and its metadata JSON. Use optional attribution or attention values and tumor filtering when generating the outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tumor_region import (  # noqa: E402
    add_tumor_region_arguments,
    apply_tumor_filter_by_case,
    remap_indices_after_mask,
    tumor_filter_kwargs,
)

from make_patch_grid import make_grid  # noqa: E402


def _patch_filename(x, y) -> str:
    """Format patch coordinates as an x/y PNG filename."""
    return f"{int(x):05d}x_{int(y):05d}y.png"


def make_polarity_labels(
    attrs: np.ndarray,
    mode: str = "top_k",
    top_k: int = 50,
    quantile: float = 0.2,
) -> np.ndarray:
    """Label attribution polarity using sign, the highest and lowest top-k values, or tail quantiles. Keep invalid values labeled none."""
    values = np.asarray(attrs, dtype=np.float64).reshape(-1)
    labels = np.full(values.shape, "none", dtype=object)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return labels

    mode = str(mode).lower()
    if mode == "sign":
        labels[finite[values[finite] > 0]] = "positive"
        labels[finite[values[finite] < 0]] = "negative"
        return labels

    if mode == "top_k":
        k = max(0, min(int(top_k), finite.size // 2))
        if k == 0:
            return labels
        order = finite[np.argsort(values[finite], kind="mergesort")]
        labels[order[-k:]] = "positive"
        labels[order[:k]] = "negative"
        return labels

    if mode == "quantile":
        q = float(quantile)
        if not (0.0 < q < 0.5):
            raise ValueError('quantile must be in (0, 0.5).')
        finite_values = values[finite]
        lo = np.quantile(finite_values, q)
        hi = np.quantile(finite_values, 1.0 - q)
        labels[finite[finite_values >= hi]] = "positive"
        labels[finite[finite_values <= lo]] = "negative"
        return labels

    raise ValueError(f"Unknown polarity mode: {mode}")


# ============================================================
# Default settings
# ============================================================

# ---------- Input / output ----------
# cohort cluster dump
DEFAULT_DUMP_INPUT = Path(
    r"outputs/interpretability/tumor_cluster/double_high/cohort_GigaPath/cohort-GigaPath-cluster_result.npz"
)
# Patch root. None uses the saved dump metadata.
DEFAULT_PATCH_ROOT: Optional[Path] = None
# Output directory. None uses the dump directory.
DEFAULT_OUT_DIR: Optional[Path] = r"outputs/interpretability/tumor_cluster/double_high/cohort_GigaPath"

# ---------- VALUE input ----------
# Both value inputs are optional.
# Integrated Gradients NPZ file or directory
DEFAULT_IG_NPZ: Optional[Path] = None
# Attention NPZ file or directory
DEFAULT_ATTENTION_NPZ: Optional[Path] = None
DEFAULT_USE_DUMP_IG = False                  # Optionally read saved dump attributions when no external IG is provided.

# ---------- Tumor filter ----------
DEFAULT_TUMOR_FILTER = True                  # Identify tumor patches using the annotations.
DEFAULT_ANNOTATION_DIR = Path(
    r"outputs/interpretability/tumor_area_output"
)                                           # Tumor annotation directory
DEFAULT_TUMOR_DISPLAY_MODE = "zero_outside"  # Draw the full tissue outline with gray outside the tumor.

# ---------- Grid ----------
DEFAULT_GRID_ROWS = 10                        # Patch grid rows
DEFAULT_GRID_COLS = 10                       # Patch grid columns
DEFAULT_PATCH_SIZE = 256                     # Rendered patch size
DEFAULT_GAP = 6                              # Gap between patches

# ---------- Slide map ----------
DEFAULT_SLIDE_MAP_ENABLED = True             # Generate per-case cluster maps.
DEFAULT_SLIDE_MAP_TILE_SIZE = 256            # Patch side length in level-0 pixels
DEFAULT_SLIDE_MAP_DOWNSAMPLE = 16            # Slide map downsample factor

# ---------- Polarity ----------
DEFAULT_RECOMPUTE_POLARITY = False           # Recalculate attribution polarity when requested.
DEFAULT_CLI_POLARITY_MODE = None             # None uses saved settings.
DEFAULT_CLI_TOP_K = None                     # Number of patches selected from each tail
DEFAULT_CLI_QUANTILE = None                  # Polarity quantile

# ---------- VALUE normalization ----------
DEFAULT_ATTR_NORM = "iqr"                    # IG per-case normalization
DEFAULT_ATTENTION_NORM = "iqr"               # Match the IG normalization convention for cross-case comparisons.
ATTR_NORM_MODES = ("none", "robust", "iqr", "zscore", "abs_scale", "rank")
_ATTR_SCALE_REL_FLOOR = 0.10                 # Minimum normalization scale
_ATTR_NORM_CLIP = 10.0                       # normalization clip [-10, 10]

# ---------- Cluster summary ----------
DEFAULT_SUMMARY_CLIP_LO = 1                # Lower percentile for robust statistics
DEFAULT_SUMMARY_CLIP_HI = 97               # Upper percentile for robust statistics
DEFAULT_SUMMARY_ROBUST_MODE = "winsor"       # {none, winsor, trim}
SUMMARY_ROBUST_MODES = ("none", "winsor", "trim")

# ---------- Outputs ----------
ALL_OUTPUTS = [
    "per_patch",
    "cluster_summary",
    "target_summary",
    "case_polarity",
    "box",
    "heatmap",
    "umap",
    "grids",
    "polarity_grids",
    "slide_map",
]
DEFAULT_OUTPUTS: List[str] = []               # An empty list generates all outputs available in the current mode.
MORPHOLOGY_ONLY_OUTPUTS = ["per_patch", "umap", "grids", "slide_map"]
VALUE_REQUIRED_OUTPUTS = {
    "per_patch",
    "cluster_summary",
    "target_summary",
    "case_polarity",
    "box",
    "heatmap",
    "polarity_grids",
}

# ---------- CLI overrides ----------
# None uses dump metadata or the module defaults.
DEFAULT_CLI_GRID_ROWS = None                  # grid rows
DEFAULT_CLI_GRID_COLS = None                  # grid cols
DEFAULT_CLI_PATCH_SIZE = None                 # patch size
DEFAULT_CLI_GAP = None                        # grid gap
DEFAULT_CLI_SEED = None                       # seed
DEFAULT_CLI_ATTR_NORM = None                  # IG normalization
DEFAULT_CLI_SUMMARY_CLIP_LO = 1            # Lower summary percentile
DEFAULT_CLI_SUMMARY_CLIP_HI = 97            # Upper summary percentile
DEFAULT_CLI_SUMMARY_ROBUST_MODE = "winsor"        # robust mode
DEFAULT_CLI_SLIDE_MAP_TILE_SIZE = None        # slide tile size
DEFAULT_CLI_SLIDE_MAP_DOWNSAMPLE = None       # slide downsample

# ---------- Reproducibility ----------
DEFAULT_SEED = 45                             # Random seed


# ============================================================
# Cohort-level statistics / plotting implementations
# ============================================================


def _safe_scale(scale: float, x: np.ndarray) -> float:
    """Protect MAD and IQR scaling from near-zero denominators."""
    sd = float(np.std(x))
    floor = _ATTR_SCALE_REL_FLOOR * sd if sd > 0 else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return floor if floor > 0 else 0.0
    return max(scale, floor)


def _attr_norm_one_case(x: np.ndarray, mode: str) -> np.ndarray:
    """Normalize one case's attribution array while preserving its shape and sign."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or mode == "none":
        return x.astype(np.float64, copy=True)

    if mode == "zscore":
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if sd <= 0 or not np.isfinite(sd):
            return np.zeros_like(x)
        return (x - mu) / sd

    if mode == "robust":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = _safe_scale(1.4826 * mad, x)
        if scale <= 0 or not np.isfinite(scale):
            return np.zeros_like(x)
        y = (x - med) / scale
        return np.clip(y, -_ATTR_NORM_CLIP, _ATTR_NORM_CLIP)

    if mode == "iqr":
        med = float(np.median(x))
        q1, q3 = np.percentile(x, [25, 75])
        iqr = float(q3 - q1)
        scale = _safe_scale(iqr / 1.349, x)
        if scale <= 0 or not np.isfinite(scale):
            return np.zeros_like(x)
        y = (x - med) / scale
        return np.clip(y, -_ATTR_NORM_CLIP, _ATTR_NORM_CLIP)

    if mode == "abs_scale":
        m = float(np.max(np.abs(x)))
        if m <= 0 or not np.isfinite(m):
            return np.zeros_like(x)
        return x / m

    if mode == "rank":
        n = x.size
        if n <= 1:
            return np.zeros_like(x)
        order = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return 2.0 * ranks / (n - 1) - 1.0

    raise ValueError(f"Unknown attribution normalization mode: {mode}")


def normalize_attrs_per_case(
    attrs_all: np.ndarray,
    per_case_info: List[dict],
    mode: str,
) -> np.ndarray:
    """Apply attribution normalization within case row spans for statistics and display. Return a new array aligned with attrs_all."""
    mode = (mode or "none").lower()
    if mode not in ATTR_NORM_MODES:
        raise ValueError(
            f"Invalid attr-norm mode: {mode}. Choose from {ATTR_NORM_MODES}"
        )
    out = np.asarray(attrs_all, dtype=np.float64).copy()
    if mode == "none":
        return out
    polarity = np.full(attrs_all.shape[0], "none", dtype=object)
    for info in per_case_info:
        s, e = int(info["start_row"]), int(info["end_row"])
        if e <= s:
            continue
        out[s:e] = _attr_norm_one_case(out[s:e], mode)
    return out


def _winsorized_mean(x: np.ndarray, clip_lo: float, clip_hi: float) -> float:
    """Clip values to the requested percentile bounds before averaging."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    lo, hi = np.percentile(x, [float(clip_lo), float(clip_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return float(np.mean(x))
    return float(np.mean(np.clip(x, lo, hi)))


def _trimmed_mean(x: np.ndarray, clip_lo: float, clip_hi: float) -> float:
    """Average only the values within the requested percentile bounds."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    lo, hi = np.percentile(x, [float(clip_lo), float(clip_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return float(np.mean(x))
    kept = x[(x >= lo) & (x <= hi)]
    if kept.size == 0:
        return float(np.mean(x))
    return float(np.mean(kept))


def _robust_mean_fields(
    x: np.ndarray,
    robust_mode: str,
    clip_lo: float,
    clip_hi: float,
) -> dict:
    """Return winsorized and trimmed means when robust statistics are enabled, and identify the primary statistic."""
    if str(robust_mode).lower() == "none":
        return {}
    return {
        "mean_attribution_winsor": _winsorized_mean(x, clip_lo, clip_hi),
        "mean_attribution_trimmed": _trimmed_mean(x, clip_lo, clip_hi),
    }


def save_per_patch_csv(
    out_csv: Path,
    case_ids_all: np.ndarray,
    target_ids_all: Optional[np.ndarray],
    target_names_all: Optional[np.ndarray],
    coords_all: np.ndarray,
    attrs_all: np.ndarray,
    cluster_id: np.ndarray,
    polarity: np.ndarray,
    patch_root: Optional[Path] = None,
    attrs_norm: Optional[np.ndarray] = None,
    attr_norm_mode: str = "none",
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    has_norm = attrs_norm is not None and str(attr_norm_mode).lower() != "none"
    norm_col = f"attribution_norm_{attr_norm_mode}" if has_norm else None
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = [
            "case", "target", "target_name", "coord_x", "coord_y",
            "attribution", "abs_attribution", "cluster", "polarity",
        ]
        if has_norm:
            header.append(norm_col)
        header.append("patch_path")
        w.writerow(header)
        for i in range(attrs_all.shape[0]):
            target_id = "" if target_ids_all is None else str(target_ids_all[i])
            target_name = "" if target_names_all is None else str(target_names_all[i])
            case = str(case_ids_all[i])
            row = [
                case,
                target_id,
                target_name,
                int(coords_all[i, 0]), int(coords_all[i, 1]),
                f"{float(attrs_all[i]):.8g}",
                f"{abs(float(attrs_all[i])):.8g}",
                int(cluster_id[i]),
                polarity[i],
            ]
            if has_norm:
                row.append(f"{float(attrs_norm[i]):.8g}")
            if patch_root is not None:
                patch_path = str(
                    patch_root / case
                    / _patch_filename(coords_all[i, 0], coords_all[i, 1])
                )
            else:
                patch_path = ""
            row.append(patch_path)
            w.writerow(row)
    print(f"[saved] {out_csv}")


def compute_cluster_summary(
    attrs_all: np.ndarray,
    cluster_id: np.ndarray,
    polarity: np.ndarray,
    n_clusters: int,
    attrs_norm: Optional[np.ndarray] = None,
    robust_mode: str = "none",
    clip_lo: float = DEFAULT_SUMMARY_CLIP_LO,
    clip_hi: float = DEFAULT_SUMMARY_CLIP_HI,
) -> List[dict]:
    """Summarize each cluster using raw attribution, optional case-normalized statistics, and optional winsorized and trimmed means."""
    stats = []
    use_norm = attrs_norm is not None
    use_robust = str(robust_mode).lower() != "none"
    for c in range(n_clusters):
        mask = cluster_id == c
        n = int(mask.sum())
        if n == 0:
            empty = {
                "cluster": c, "n": 0,
                "mean_attribution": 0.0, "median_attribution": 0.0,
                "mean_abs_attribution": 0.0,
                "n_positive": 0, "n_negative": 0, "n_none": 0,
                "positive_ratio": 0.0, "negative_ratio": 0.0,
            }
            if use_norm:
                empty["mean_attribution_norm"] = 0.0
                empty["median_attribution_norm"] = 0.0
                empty["mean_abs_attribution_norm"] = 0.0
            if use_robust:
                empty["mean_attribution_winsor"] = 0.0
                empty["mean_attribution_trimmed"] = 0.0
            stats.append(empty)
            continue
        a = attrs_all[mask]
        n_pos = int((polarity[mask] == "positive").sum())
        n_neg = int((polarity[mask] == "negative").sum())
        n_none = int((polarity[mask] == "none").sum())
        entry = {
            "cluster": c, "n": n,
            "mean_attribution": float(np.mean(a)),
            "median_attribution": float(np.median(a)),
            "mean_abs_attribution": float(np.mean(np.abs(a))),
            "n_positive": n_pos, "n_negative": n_neg, "n_none": n_none,
            "positive_ratio": n_pos / n, "negative_ratio": n_neg / n,
        }
        if use_norm:
            an = attrs_norm[mask]
            entry["mean_attribution_norm"] = float(np.mean(an))
            entry["median_attribution_norm"] = float(np.median(an))
            entry["mean_abs_attribution_norm"] = float(np.mean(np.abs(an)))
        if use_robust:
            entry.update(_robust_mean_fields(a, robust_mode, clip_lo, clip_hi))
        stats.append(entry)
    return stats


def save_cluster_summary_csv(out_csv: Path, stats: List[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        for r in stats:
            w.writerow(r)
    print(f"[saved] {out_csv}")


def compute_case_cluster_polarity_table(
    case_ids_all: np.ndarray,
    cluster_id: np.ndarray,
    polarity: np.ndarray,
    n_clusters: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return ordered cases and matrices of total, positive, and negative tile counts by case and cluster."""
    cases = sorted(set(str(c) for c in case_ids_all))
    n_case = len(cases)
    count_total = np.zeros((n_case, n_clusters), dtype=np.int64)
    count_pos = np.zeros((n_case, n_clusters), dtype=np.int64)
    count_neg = np.zeros((n_case, n_clusters), dtype=np.int64)

    case_idx = {c: i for i, c in enumerate(cases)}

    for i in range(cluster_id.shape[0]):
        ci = case_idx[str(case_ids_all[i])]
        cj = int(cluster_id[i])
        if cj < 0 or cj >= n_clusters:
            continue
        count_total[ci, cj] += 1
        if polarity[i] == "positive":
            count_pos[ci, cj] += 1
        elif polarity[i] == "negative":
            count_neg[ci, cj] += 1

    return cases, count_total, count_pos, count_neg


def save_case_cluster_polarity_csv(
    out_csv: Path,
    cases: List[str],
    count_total: np.ndarray,
    count_pos: np.ndarray,
    count_neg: np.ndarray,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_clusters = count_total.shape[1]
    fields = ["case"]
    for c in range(n_clusters):
        fields += [f"n_total_c{c}", f"n_positive_c{c}", f"n_negative_c{c}",
                   f"positive_ratio_c{c}", f"negative_ratio_c{c}"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, case in enumerate(cases):
            row = {"case": case}
            for c in range(n_clusters):
                n_t = int(count_total[i, c])
                n_p = int(count_pos[i, c])
                n_n = int(count_neg[i, c])
                row[f"n_total_c{c}"] = n_t
                row[f"n_positive_c{c}"] = n_p
                row[f"n_negative_c{c}"] = n_n
                row[f"positive_ratio_c{c}"] = (n_p / n_t) if n_t > 0 else 0.0
                row[f"negative_ratio_c{c}"] = (n_n / n_t) if n_t > 0 else 0.0
            w.writerow(row)
    print(f"[saved] {out_csv}")


def _target_display_key(target_id: str, target_name: str) -> str:
    return target_name or target_id or "UNKNOWN"


def _add_rank_fields(rows: List[dict], key: str, pos_field: str, neg_field: str) -> None:
    valid = [r for r in rows if int(r.get("n", 0)) > 0 and key in r]
    for rank, r in enumerate(
        sorted(valid, key=lambda x: float(x[key]), reverse=True),
        start=1,
    ):
        r[pos_field] = rank
    for rank, r in enumerate(
        sorted(valid, key=lambda x: float(x[key])),
        start=1,
    ):
        r[neg_field] = rank
    for r in rows:
        r.setdefault(pos_field, "")
        r.setdefault(neg_field, "")


def compute_target_cluster_summary(
    attrs_all: np.ndarray,
    cluster_id: np.ndarray,
    polarity: np.ndarray,
    target_ids_all: np.ndarray,
    target_names_all: np.ndarray,
    n_clusters: int,
    attrs_norm: Optional[np.ndarray] = None,
    robust_mode: str = "none",
    clip_lo: float = DEFAULT_SUMMARY_CLIP_LO,
    clip_hi: float = DEFAULT_SUMMARY_CLIP_HI,
) -> List[dict]:
    """Summarize each cluster by target class and rank the means in both directions. Include robust means and rankings when enabled."""
    use_norm = attrs_norm is not None
    use_robust = str(robust_mode).lower() != "none"
    target_pairs = sorted({
        (str(tid), str(tname))
        for tid, tname in zip(target_ids_all, target_names_all)
    }, key=lambda x: (_target_display_key(x[0], x[1]), x[0]))

    all_rows: List[dict] = []
    for target_id, target_name in target_pairs:
        target_mask = (target_ids_all.astype(str) == target_id) & (
            target_names_all.astype(str) == target_name
        )
        target_rows: List[dict] = []
        n_target_patch = int(target_mask.sum())
        for c in range(n_clusters):
            mask = target_mask & (cluster_id == c)
            n = int(mask.sum())
            if n == 0:
                row = {
                    "target": target_id,
                    "target_name": target_name,
                    "cluster": c,
                    "cluster_label": f"c{c}",
                    "n": 0,
                    "n_target_patch": n_target_patch,
                    "mean_attribution": 0.0,
                    "median_attribution": 0.0,
                    "mean_abs_attribution": 0.0,
                    "n_positive": 0,
                    "n_negative": 0,
                    "n_none": 0,
                    "positive_ratio": 0.0,
                    "negative_ratio": 0.0,
                }
                if use_norm:
                    row["mean_attribution_norm"] = 0.0
                    row["median_attribution_norm"] = 0.0
                    row["mean_abs_attribution_norm"] = 0.0
                if use_robust:
                    row["mean_attribution_winsor"] = 0.0
                    row["mean_attribution_trimmed"] = 0.0
                target_rows.append(row)
                continue

            a = attrs_all[mask]
            n_pos = int((polarity[mask] == "positive").sum())
            n_neg = int((polarity[mask] == "negative").sum())
            n_none = int((polarity[mask] == "none").sum())
            row = {
                "target": target_id,
                "target_name": target_name,
                "cluster": c,
                "cluster_label": f"c{c}",
                "n": n,
                "n_target_patch": n_target_patch,
                "mean_attribution": float(np.mean(a)),
                "median_attribution": float(np.median(a)),
                "mean_abs_attribution": float(np.mean(np.abs(a))),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_none": n_none,
                "positive_ratio": n_pos / n,
                "negative_ratio": n_neg / n,
            }
            if use_norm:
                an = attrs_norm[mask]
                row["mean_attribution_norm"] = float(np.mean(an))
                row["median_attribution_norm"] = float(np.median(an))
                row["mean_abs_attribution_norm"] = float(np.mean(np.abs(an)))
            if use_robust:
                row.update(_robust_mean_fields(a, robust_mode, clip_lo, clip_hi))
            target_rows.append(row)

        _add_rank_fields(
            target_rows,
            key="mean_attribution",
            pos_field="positive_rank_mean_attribution",
            neg_field="negative_rank_mean_attribution",
        )
        if use_norm:
            _add_rank_fields(
                target_rows,
                key="mean_attribution_norm",
                pos_field="positive_rank_mean_attribution_norm",
                neg_field="negative_rank_mean_attribution_norm",
            )
        if use_robust:
            _add_rank_fields(
                target_rows,
                key="mean_attribution_winsor",
                pos_field="positive_rank_mean_attribution_winsor",
                neg_field="negative_rank_mean_attribution_winsor",
            )
            _add_rank_fields(
                target_rows,
                key="mean_attribution_trimmed",
                pos_field="positive_rank_mean_attribution_trimmed",
                neg_field="negative_rank_mean_attribution_trimmed",
            )
        all_rows.extend(target_rows)

    return all_rows


def save_target_cluster_rank_order_csv(
    out_csv: Path, rows: List[dict], target_source: str = "",
) -> None:
    """Write class-specific cluster rankings with the grouping source. A pred_class source identifies model prediction as the grouping field."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict = {}
    for r in rows:
        key = (str(r.get("target", "")), str(r.get("target_name", "")))
        grouped.setdefault(key, []).append(r)

    fields = [
        "target", "target_name", "rank_metric",
        "positive_order", "negative_order", "target_source",
    ]
    has_norm = any("mean_attribution_norm" in r for r in rows)
    has_winsor = any("mean_attribution_winsor" in r for r in rows)
    has_trimmed = any("mean_attribution_trimmed" in r for r in rows)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (target_id, target_name), target_rows in grouped.items():
            metrics = ["mean_attribution"]
            if has_norm:
                metrics.append("mean_attribution_norm")
            if has_winsor:
                metrics.append("mean_attribution_winsor")
            if has_trimmed:
                metrics.append("mean_attribution_trimmed")
            for metric in metrics:
                valid = [r for r in target_rows if int(r.get("n", 0)) > 0]
                pos = sorted(valid, key=lambda r: float(r[metric]), reverse=True)
                neg = sorted(valid, key=lambda r: float(r[metric]))
                w.writerow({
                    "target": target_id,
                    "target_name": target_name,
                    "rank_metric": metric,
                    "positive_order": " > ".join(
                        f"{r['cluster_label']}({float(r[metric]):.4g})" for r in pos
                    ),
                    "negative_order": " > ".join(
                        f"{r['cluster_label']}({float(r[metric]):.4g})" for r in neg
                    ),
                    "target_source": target_source,
                })
    print(f"[saved] {out_csv}")


def save_umap_scatter(
    out_path_png: Path,
    embedding: np.ndarray,
    cluster_id: np.ndarray,
    n_clusters: int,
    title: str,
    indices: Optional[np.ndarray] = None,
    seed: int = 42,
    point_size: float = 4.0,
    alpha: float = 0.55,
    figsize: Tuple[float, float] = (8.0, 7.0),
    show_labels: bool = True,
    label_fontsize: int = 10,
    show_arrow_axes: bool = True,
    palette: Optional[str] = None,
    bottom_legend: bool = False,
) -> None:
    """Plot a two-dimensional UMAP embedding with cluster colors and labels. An optional index array aligns sampled points and clusters."""
    if bottom_legend:
        show_labels = False
    if indices is not None:
        cluster_id = cluster_id[indices]

    cmap_name = palette or ("tab10" if n_clusters <= 10 else "tab20")
    cmap = plt.get_cmap(cmap_name)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    order = rng.permutation(embedding.shape[0])
    colors = np.array([cmap(int(c) % cmap.N) for c in cluster_id[order]])
    ax.scatter(
        embedding[order, 0], embedding[order, 1],
        s=float(point_size), c=colors, alpha=float(alpha), linewidths=0,
    )

    if show_labels:
        for c in range(n_clusters):
            mask = cluster_id == c
            if not np.any(mask):
                continue
            cx = float(np.median(embedding[mask, 0]))
            cy = float(np.median(embedding[mask, 1]))
            ax.text(
                cx, cy, str(c),
                ha="center", va="center",
                fontsize=int(label_fontsize), color="#222",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white", edgecolor="#555", linewidth=0.9,
                    alpha=0.95,
                ),
            )

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    if show_arrow_axes:
        arrow_props = dict(
            arrowstyle="-|>", color="#333", lw=1.4, mutation_scale=14,
        )
        ax.annotate(
            "", xy=(0.12, 0.04), xytext=(0.02, 0.04),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_props,
        )
        ax.annotate(
            "", xy=(0.02, 0.14), xytext=(0.02, 0.04),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=arrow_props,
        )
        ax.text(
            0.13, 0.025, "UMAP1",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=int(label_fontsize), color="#333",
        )
        ax.text(
            0.025, 0.15, "UMAP2",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=int(label_fontsize), color="#333", rotation=90,
        )

    if title:
        ax.set_title(title, fontsize=11)

    if bottom_legend:
        present_ids = np.unique(cluster_id)
        handles = []
        for c in present_ids.tolist():
            c = int(c)
            if c < 0 or c >= n_clusters:
                continue
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o", color="none",
                    markerfacecolor=cmap(c % cmap.N),
                    markeredgecolor="none",
                    markersize=9,
                    label=f"C{c}",
                )
            )
        if handles:
            max_per_row = 12
            ncol = min(max_per_row, len(handles))
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=ncol,
                fontsize=9,
                frameon=False,
                handletextpad=0.4,
                columnspacing=1.2,
                borderaxespad=0.0,
            )

    fig.tight_layout()
    if bottom_legend:
        fig.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(out_path_png.with_suffix(".pdf"),
                    bbox_inches="tight", pad_inches=0.12)
    else:
        fig.savefig(out_path_png, dpi=300)
        fig.savefig(out_path_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {out_path_png}")


def save_cluster_boxplot(
    out_path_png: Path,
    attrs_all: np.ndarray,
    cluster_id: np.ndarray,
    n_clusters: int,
    title: str,
    ylabel: str = "IG attribution",
    highlight_positive: Optional[int] = None,
    highlight_negative: Optional[int] = None,
    sort_key_label: Optional[str] = None,
) -> None:
    """Plot the attribution distribution per cluster and mark medians, means, and outliers."""
    data = [attrs_all[cluster_id == c] for c in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 1.2), 4.8), dpi=200)
    bp = ax.boxplot(
        data,
        labels=[f"c{c}" for c in range(n_clusters)],
        showfliers=True,
        flierprops=dict(
            marker="o",
            markersize=2.0,
            markerfacecolor="#999999",
            markeredgecolor="none",
            alpha=0.35,
        ),
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="#2CA02C",
            markeredgecolor="#1F6B1F",
            markersize=6,
        ),
        medianprops=dict(color="#FF7F0E", linewidth=1.4),
        whiskerprops=dict(color="#444444", linewidth=1.0),
        capprops=dict(color="#444444", linewidth=1.0),
        boxprops=dict(color="#333333", linewidth=1.0),
    )

    ax.axhline(0, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    y_lows: List[float] = []
    y_highs: List[float] = []
    for d in data:
        if d.size == 0:
            continue
        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        in_range = d[(d >= lo) & (d <= hi)]
        if in_range.size > 0:
            y_lows.append(float(in_range.min()))
            y_highs.append(float(in_range.max()))
        else:
            y_lows.append(float(q1))
            y_highs.append(float(q3))
    if y_lows and y_highs:
        y_lo = min(y_lows)
        y_hi = max(y_highs)
        if not np.isfinite(y_lo) or not np.isfinite(y_hi) or y_hi <= y_lo:
            pool = np.concatenate([d for d in data if d.size > 0])
            if pool.size > 0:
                y_lo = float(np.percentile(pool, 1))
                y_hi = float(np.percentile(pool, 99))
        pad = (y_hi - y_lo) * 0.1 if y_hi > y_lo else 1.0
        ax.set_ylim(y_lo - pad, y_hi + pad)

    legend_handles = [
        plt.Line2D([0], [0], marker="D", color="none",
                   markerfacecolor="#2CA02C", markeredgecolor="#1F6B1F",
                   markersize=7, label="mean"),
        plt.Line2D([0], [0], color="#FF7F0E", linewidth=2, label="median"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="#999999", markeredgecolor="none",
                   markersize=4, alpha=0.6, label="outliers"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.9,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {out_path_png}")


def save_case_cluster_heatmap(
    out_path_png: Path,
    cases: List[str],
    count_total: np.ndarray,
    count_pos: np.ndarray,
    count_neg: np.ndarray,
    title: str,
) -> None:
    """Plot positive minus negative tile fractions on a case-by-cluster grid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_r = np.where(count_total > 0, count_pos / np.maximum(count_total, 1), 0.0)
        neg_r = np.where(count_total > 0, count_neg / np.maximum(count_total, 1), 0.0)
    score = pos_r - neg_r

    n_case, n_clusters = score.shape
    fig_w = max(6.0, 0.8 + 0.7 * n_clusters)
    fig_h = max(3.0, 0.4 + 0.3 * n_case)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    vmax = max(abs(score).max(), 1e-6)
    im = ax.imshow(score, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"c{c}" for c in range(n_clusters)])
    ax.set_yticks(range(n_case))
    ax.set_yticklabels(cases, fontsize=8)
    ax.set_xlabel("cluster")
    ax.set_title(title)

    for i in range(n_case):
        for j in range(n_clusters):
            txt = f"{score[i, j]:+.2f}\nn={int(count_total[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="black" if abs(score[i, j]) < 0.5 * vmax else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("positive_ratio − negative_ratio")
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[saved] {out_path_png}")


def cluster_color_palette(n_clusters: int) -> List[tuple]:
    """Return one RGBA color per cluster using the same palette as the embedding plots."""
    cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    return [cmap(c % cmap.N) for c in range(n_clusters)]


def save_case_cluster_slide_map(
    out_path_png: Path,
    case: str,
    cluster_id_case: np.ndarray,
    coords_case: np.ndarray,
    n_clusters: int,
    palette: List[tuple],
    tile_size: int,
    downsample: int = 16,
    title: Optional[str] = None,
    dot_fill_ratio: float = 0.78,
) -> None:
    """Draw tile positions as colored dots for one slide. Use gray for negative cluster IDs."""
    coords = np.asarray(coords_case, dtype=np.int64)
    cluster_ids = np.asarray(cluster_id_case)
    if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] == 0:
        print(f"[skip] Case {case} has no coordinates for a slide map.")
        return

    step = max(int(tile_size), 1)
    cols = ((coords[:, 0] - coords[:, 0].min()) // step).astype(np.int64)
    rows = ((coords[:, 1] - coords[:, 1].min()) // step).astype(np.int64)
    n_cols = int(cols.max()) + 1
    n_rows = int(rows.max()) + 1

    n_palette = len(palette)
    colors = np.empty((cluster_ids.shape[0], 4), dtype=np.float32)
    fallback = np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float32)
    for i, c in enumerate(cluster_ids.tolist()):
        c = int(c)
        if 0 <= c < n_palette:
            colors[i] = np.asarray(palette[c], dtype=np.float32)
        else:
            colors[i] = fallback

    base = 9.0
    if n_cols >= n_rows:
        fig_w = base
        fig_h = max(3.0, base * (n_rows / max(n_cols, 1)))
    else:
        fig_h = base
        fig_w = max(3.0, base * (n_cols / max(n_rows, 1)))
    fig_w += 0.6
    fig_h += 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_axis_off()

    cell_inch = min(fig_w / max(n_cols, 1), fig_h / max(n_rows, 1)) * 0.92
    cell_pt = cell_inch * 72.0
    marker_diam_pt = cell_pt * float(dot_fill_ratio)
    marker_size = max(4.0, marker_diam_pt ** 2)

    ax.scatter(
        cols.astype(np.float32),
        rows.astype(np.float32),
        s=marker_size,
        c=colors,
        marker="o",
        linewidths=0,
    )

    if title is None:
        title = (
            f"{case}  cluster slide map  "
            f"(n_patch={int(coords.shape[0])}, K={int(n_clusters)}, "
            f"grid={n_rows}×{n_cols})"
        )
    ax.set_title(title, fontsize=11)

    present_ids, counts = np.unique(cluster_ids, return_counts=True)
    handles = []
    for c in present_ids.tolist():
        c = int(c)
        if c < 0 or c >= n_palette:
            continue
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o", color="none",
                markerfacecolor=palette[c],
                markeredgecolor="none",
                markersize=9,
                label=f"C{c}",
            )
        )
    if handles:
        max_per_row = 12
        n_legend = len(handles)
        ncol = min(max_per_row, n_legend) if n_legend else 1
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            fontsize=9,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.2,
            borderaxespad=0.0,
        )

    fig.tight_layout()
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path_png.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[saved] {out_path_png}")


def save_cohort_case_cluster_maps(
    out_dir: Path,
    cohort_prefix: str,
    cluster_id: np.ndarray,
    coords_all: np.ndarray,
    case_ids_all: np.ndarray,
    per_case_info: List[dict],
    n_clusters: int,
    tile_size: int,
    downsample: int,
) -> None:
    """Write per-case tile maps with a shared cluster palette."""
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = cluster_color_palette(n_clusters)

    for info in per_case_info:
        case = str(info["case"])
        s, e = int(info["start_row"]), int(info["end_row"])
        if e <= s:
            continue
        cluster_id_case = cluster_id[s:e]
        coords_case = coords_all[s:e]
        if coords_case.size == 0:
            continue
        out_path = out_dir / f"{cohort_prefix}-case_{case}-cluster_map.png"
        try:
            save_case_cluster_slide_map(
                out_path_png=out_path,
                case=case,
                cluster_id_case=cluster_id_case,
                coords_case=coords_case,
                n_clusters=n_clusters,
                palette=palette,
                tile_size=tile_size,
                downsample=downsample,
            )
        except Exception as e:
            print(f"[warning] Case {case} could not render its cluster slide map: {e}")


def collect_global_cluster_patch_records(
    cluster_id: np.ndarray,
    cluster_target: int,
    case_ids_all: np.ndarray,
    coords_all: np.ndarray,
    patch_root: Path,
    max_paths: int,
    seed: int,
) -> Tuple[List[dict], dict]:
    """Collect candidate examples from one global cluster, balancing cases where possible."""
    rng = np.random.default_rng(seed)
    idx = np.where(cluster_id == cluster_target)[0]
    if idx.size == 0:
        return [], {}

    cases_in_cluster: dict = {}
    for i in idx:
        c = str(case_ids_all[i])
        cases_in_cluster.setdefault(c, []).append(int(i))

    n_case = len(cases_in_cluster)
    per_case_quota = max(1, (max_paths + n_case - 1) // n_case)

    picked: List[int] = []
    contribution: dict = {}
    for case, members in cases_in_cluster.items():
        take = min(per_case_quota, len(members))
        sample = rng.choice(members, take, replace=False).tolist()
        picked.extend(sample)
        contribution[case] = take

    rng.shuffle(picked)
    picked = picked[:max_paths * 3]

    records: List[dict] = []
    missing = 0
    for i in picked:
        case = str(case_ids_all[i])
        fname = _patch_filename(coords_all[i, 0], coords_all[i, 1])
        p = patch_root / case / fname
        if p.exists():
            records.append({
                "cohort_patch_id": int(i),
                "cluster": int(cluster_target),
                "case": case,
                "coord_x": int(coords_all[i, 0]),
                "coord_y": int(coords_all[i, 1]),
                "patch_filename": fname,
                "patch_path": str(p),
            })
        else:
            missing += 1
        if len(records) >= max_paths:
            break

    if missing > 0:
        print(f"  [warning] Cluster {cluster_target} has {missing} missing patch images.")

    return records, contribution


def collect_global_cluster_patch_paths(
    cluster_id: np.ndarray,
    cluster_target: int,
    case_ids_all: np.ndarray,
    coords_all: np.ndarray,
    patch_root: Path,
    max_paths: int,
    seed: int,
) -> Tuple[List[Path], dict]:
    records, contribution = collect_global_cluster_patch_records(
        cluster_id=cluster_id,
        cluster_target=cluster_target,
        case_ids_all=case_ids_all,
        coords_all=coords_all,
        patch_root=patch_root,
        max_paths=max_paths,
        seed=seed,
    )
    return [Path(r["patch_path"]) for r in records], contribution


def _case_start_map(per_case_info: List[dict]) -> dict:
    return {str(c["case"]): int(c["start_row"]) for c in per_case_info}


def _select_grid_records(
    records: List[dict],
    need_n: int,
    grid_cols: int,
    seed: int,
    case_start: dict,
    attrs_all: np.ndarray,
    polarity: np.ndarray,
) -> List[dict]:
    """Fix the patch order and assign grid row and column positions."""
    chosen = random.Random(seed).sample(records, need_n)
    out: List[dict] = []
    for pos, rec in enumerate(chosen):
        row0 = pos // grid_cols
        col0 = pos % grid_cols
        cohort_patch_id = int(rec["cohort_patch_id"])
        case = str(rec["case"])
        item = dict(rec)
        item.update({
            "grid_index": int(pos),
            "grid_row": int(row0 + 1),
            "grid_col": int(col0 + 1),
            "grid_row0": int(row0),
            "grid_col0": int(col0),
            "case_patch_id": int(cohort_patch_id - case_start.get(case, 0)),
            "attribution": float(attrs_all[cohort_patch_id]),
            "polarity": str(polarity[cohort_patch_id]),
        })
        out.append(item)
    return out


def _best_grid_shape_for_available(
    n_available: int,
    max_rows: int,
    max_cols: int,
) -> Tuple[int, int, int]:
    """Find the largest grid supported by the available patches within the requested dimensions."""
    n_available = int(n_available)
    max_rows = max(1, int(max_rows))
    max_cols = max(1, int(max_cols))
    if n_available <= 0:
        return 0, 0, 0

    best_rows, best_cols, best_n = 1, 1, 1
    target_aspect = max_cols / max_rows
    best_aspect_delta = float("inf")
    for rows in range(1, max_rows + 1):
        for cols in range(1, max_cols + 1):
            n = rows * cols
            if n > n_available:
                continue
            aspect_delta = abs((cols / rows) - target_aspect)
            if n > best_n or (n == best_n and aspect_delta < best_aspect_delta):
                best_rows, best_cols, best_n = rows, cols, n
                best_aspect_delta = aspect_delta
    return best_rows, best_cols, best_n


def save_grid_patch_info(
    output_prefix: Path,
    cohort_prefix: str,
    cluster: int,
    grid_rows: int,
    grid_cols: int,
    selected_records: List[dict],
) -> None:
    csv_path = output_prefix.with_name(f"{output_prefix.name}_patch_info").with_suffix(".csv")
    json_path = output_prefix.with_name(f"{output_prefix.name}_patch_info").with_suffix(".json")

    fields = [
        "grid_index",
        "grid_row",
        "grid_col",
        "grid_row0",
        "grid_col0",
        "cohort_patch_id",
        "case_patch_id",
        "case",
        "cluster",
        "coord_x",
        "coord_y",
        "patch_filename",
        "patch_path",
        "attribution",
        "polarity",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in selected_records:
            w.writerow({k: rec.get(k, "") for k in fields})

    meta = {
        "schema_version": 1,
        "cohort_prefix": cohort_prefix,
        "cluster": int(cluster),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "index_note": 'grid_row and grid_col are one-based. grid_row0 and grid_col0 are zero-based.',
        "patches": selected_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[saved] {csv_path}")
    print(f"[saved] {json_path}")


def save_morphology_per_patch_csv(out_csv, case_ids_all, coords_all, cluster_id, patch_root):
    """Export coordinate, cluster and patch path for every morphology patch."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cohort_patch_id", "case", "coord_x", "coord_y", "cluster", "patch_path"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(len(cluster_id)):
            case = str(case_ids_all[i]); x, y = int(coords_all[i, 0]), int(coords_all[i, 1])
            path = "" if patch_root is None else str(patch_root / case / _patch_filename(x, y))
            w.writerow({"cohort_patch_id": i, "case": case, "coord_x": x, "coord_y": y,
                        "cluster": int(cluster_id[i]), "patch_path": path})
    print(f"[saved] {out_csv}")


def save_global_cluster_grids(
    grids_dir: Path,
    cohort_prefix: str,
    cluster_stats: List[dict],
    cluster_id: np.ndarray,
    case_ids_all: np.ndarray,
    coords_all: np.ndarray,
    attrs_all: np.ndarray,
    polarity: np.ndarray,
    per_case_info: List[dict],
    patch_root: Path,
    grid_rows: int,
    grid_cols: int,
    patch_size: int,
    gap: int,
    seed: int,
    attr_norm_mode: str = "none",
) -> None:
    grids_dir.mkdir(parents=True, exist_ok=True)
    stale_files = [
        p for p in grids_dir.glob(f"{cohort_prefix}-global_cluster_*")
        if p.is_file()
    ]
    for p in stale_files:
        try:
            p.unlink()
        except OSError as e:
            print(f"[warning] Could not remove previous global cluster grid file {p}: {e}")
    if stale_files:
        print(f"[cleanup] Removed previous global cluster grid files {len(stale_files)} files")

    need_n = grid_rows * grid_cols
    case_start = _case_start_map(per_case_info)
    norm_active = str(attr_norm_mode).lower() != "none"

    palette = [
        "#B7E1A1", "#F4B6B6", "#B6D7F4", "#F4D6A6",
        "#D6B6F4", "#B6F4E8", "#F4A6B6", "#A6E1B6",
    ]

    for stat in cluster_stats:
        c = stat["cluster"]
        if stat["n"] == 0:
            continue

        records, contrib = collect_global_cluster_patch_records(
            cluster_id=cluster_id,
            cluster_target=c,
            case_ids_all=case_ids_all,
            coords_all=coords_all,
            patch_root=patch_root,
            max_paths=need_n * 4,
            seed=seed,
        )
        grid_rows_eff, grid_cols_eff, need_n_eff = _best_grid_shape_for_available(
            n_available=len(records),
            max_rows=grid_rows,
            max_cols=grid_cols,
        )
        if need_n_eff <= 0:
            print(f"[skip] Global cluster {c} has no available patch images.")
            continue
        if need_n_eff < need_n:
            print(
                f"[info] Global cluster {c} has fewer patches than the default grid ({grid_rows}×{grid_cols} "
                f"({len(records)} < {need_n}). Using "
                f"{grid_rows_eff}×{grid_cols_eff}"
            )

        selected_records = _select_grid_records(
            records=records,
            need_n=need_n_eff,
            grid_cols=grid_cols_eff,
            seed=seed,
            case_start=case_start,
            attrs_all=attrs_all,
            polarity=polarity,
        )
        selected_paths = [Path(r["patch_path"]) for r in selected_records]
        output_prefix = grids_dir / f"{cohort_prefix}-global_cluster_{c}"

        title_parts = [
            f"Global Cluster {c}",
            f"n={stat['n']}",
            f"mean_attr={stat['mean_attribution']:+.3e}",
        ]
        if norm_active and "mean_attribution_norm" in stat:
            title_parts.append(
                f"mean_attr_{attr_norm_mode}="
                f"{stat['mean_attribution_norm']:+.2f}"
            )
        title_parts.extend([
            f"pos={stat['n_positive']}",
            f"neg={stat['n_negative']}",
            f"cases={len(contrib)}",
        ])
        title = "  ".join(title_parts)

        try:
            make_grid(
                image_paths=selected_paths,
                output_prefix=output_prefix,
                title=title,
                title_bg=palette[c % len(palette)],
                title_color="#F0F0F0",
                font_size=22,
                patch_size=patch_size,
                gap=gap,
                grid_rows=grid_rows_eff,
                grid_cols=grid_cols_eff,
                seed=seed,
                random_sample=False,
            )
            save_grid_patch_info(
                output_prefix=output_prefix,
                cohort_prefix=cohort_prefix,
                cluster=c,
                grid_rows=grid_rows_eff,
                grid_cols=grid_cols_eff,
                selected_records=selected_records,
            )
        except ValueError as e:
            print(f"[warning] Global cluster {c} grid could not be rendered: {e}")


def save_polarity_representative_grids(
    out_dir: Path,
    cohort_prefix: str,
    cluster_stats: List[dict],
    cluster_id: np.ndarray,
    case_ids_all: np.ndarray,
    coords_all: np.ndarray,
    patch_root: Path,
    grid_rows: int,
    grid_cols: int,
    patch_size: int,
    gap: int,
    seed: int,
    attr_norm_mode: str = "none",
) -> None:
    """Export grids for the clusters with the highest and lowest mean attribution. Prefer case-normalized values when available."""
    valid = [s for s in cluster_stats if s["n"] > 0]
    if not valid:
        return

    norm_active = (
        str(attr_norm_mode).lower() != "none"
        and "mean_attribution_norm" in valid[0]
    )
    sort_key = "mean_attribution_norm" if norm_active else "mean_attribution"
    pos_stat = max(valid, key=lambda s: s[sort_key])
    neg_stat = min(valid, key=lambda s: s[sort_key])
    print(
        f"[polarity representatives] Ranking field: "
        f"{'mean_attribution_norm (' + attr_norm_mode + ')' if norm_active else 'mean_attribution (raw)'}  "
        f"positive=cluster {pos_stat['cluster']}  "
        f"negative=cluster {neg_stat['cluster']}"
    )

    need_n = grid_rows * grid_cols

    for kind, stat, bg in [
        ("positive_cluster", pos_stat, "#F4B6B6"),
        ("negative_cluster", neg_stat, "#B6D7F4"),
    ]:
        c = stat["cluster"]
        paths, contrib = collect_global_cluster_patch_paths(
            cluster_id=cluster_id,
            cluster_target=c,
            case_ids_all=case_ids_all,
            coords_all=coords_all,
            patch_root=patch_root,
            max_paths=need_n * 4,
            seed=seed,
        )
        if len(paths) < need_n:
            print(f"[skip] {kind} (cluster {c}) has too few available patches.")
            continue

        label_zh = "Positive" if kind == "positive_cluster" else "Negative"
        title = f"{label_zh}  (global cluster {c})"
        try:
            make_grid(
                image_paths=paths,
                output_prefix=out_dir / f"{cohort_prefix}-{kind}_grid",
                title=title,
                title_bg=bg,
                title_color="#F0F0F0",
                patch_size=patch_size,
                gap=gap,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                seed=seed,
                random_sample=True,
            )
        except ValueError as e:
            print(f"[warning] {kind} grid could not be rendered: {e}")


def _strip_suffix(name: str) -> Optional[str]:
    for suf in ("-cluster_result.npz", "-cluster_result.json"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


def find_dump_pairs(dump_input: Path) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []

    if dump_input.is_file():
        prefix = _strip_suffix(dump_input.name)
        if prefix is None:
            raise ValueError(
                f"Expected a dump ending in -cluster_result.npz or -cluster_result.json: {dump_input}"
            )
        npz_path = dump_input.with_name(f"{prefix}-cluster_result.npz")
        json_path = dump_input.with_name(f"{prefix}-cluster_result.json")
        pairs.append((npz_path, json_path, prefix))
        return pairs

    if not dump_input.is_dir():
        raise FileNotFoundError(f"Invalid dump path: {dump_input}")

    npz_files = sorted(dump_input.rglob("*-cluster_result.npz"))
    for npz_path in npz_files:
        prefix = _strip_suffix(npz_path.name)
        if prefix is None:
            continue
        json_path = npz_path.with_name(f"{prefix}-cluster_result.json")
        pairs.append((npz_path, json_path, prefix))
    return pairs


def load_dump(npz_path: Path, json_path: Path) -> Tuple[dict, dict]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Dump NPZ was not found: {npz_path}")
    arrays = dict(np.load(npz_path, allow_pickle=True))
    meta: dict = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        print(f"[warning] Dump metadata JSON missing. Using the NPZ alone: {json_path}")
    return arrays, meta


def optional_path(text: str) -> Optional[Path]:
    """Treat none or an empty path as a disabled value input."""
    if text is None or str(text).strip().lower() in {"", "none", "null"}:
        return None
    return Path(text)


def iter_value_npz(value_input: Path) -> List[Path]:
    if value_input.is_file():
        return [value_input]
    if value_input.is_dir():
        return sorted(value_input.rglob("*.npz"))
    raise FileNotFoundError(f"Invalid value NPZ path: {value_input}")


def infer_value_case_id(path: Path, arrays: dict) -> str:
    for key in ("case_id", "case"):
        if key in arrays:
            value = np.asarray(arrays[key]).reshape(-1)
            if value.size:
                item = value[0]
                if isinstance(item, bytes):
                    item = item.decode("utf-8", errors="replace")
                return str(item)
    if path.stem in {"attributions", "attention", "attentions"}:
        return path.parent.name
    return path.stem.split("-")[0]


def _npz_scalar_str(arrays: dict, key: str) -> str:
    if key not in arrays:
        return ""
    value = np.asarray(arrays[key]).reshape(-1)
    if value.size == 0:
        return ""
    item = value[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    return str(item)


def _resolve_class_name(arrays: dict, class_id: str, name_key: str) -> str:
    """Read the stored class name or look it up in class_names using the class ID."""
    name = _npz_scalar_str(arrays, name_key)
    if name or not class_id or "class_names" not in arrays:
        return name
    class_names = np.asarray(arrays["class_names"]).reshape(-1)
    try:
        index = int(float(class_id))
    except ValueError:
        return ""
    if not (0 <= index < class_names.size):
        return ""
    item = class_names[index]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    return str(item)


def infer_value_target(arrays: dict) -> Tuple[str, str]:
    """Read the attribution target class from the NPZ."""
    target_id = _npz_scalar_str(arrays, "target_class")
    return target_id, _resolve_class_name(arrays, target_id, "target_name")


def infer_value_pred(arrays: dict) -> Tuple[str, str]:
    """Read the predicted class from the NPZ as a grouping fallback."""
    pred_id = _npz_scalar_str(arrays, "pred_class")
    return pred_id, _resolve_class_name(arrays, pred_id, "pred_name")


def build_target_arrays_from_case_map(
    case_ids_all: np.ndarray,
    case_targets: Dict[str, Tuple[str, str]],
    source_label: str = "target_class",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Expand per-case target IDs and names into arrays aligned with the tiles."""
    if not case_targets:
        return None
    case_str = np.asarray(case_ids_all, dtype=object).astype(str)
    target_ids = np.array(
        [case_targets.get(c, ("", ""))[0] for c in case_str], dtype=object
    )
    target_names = np.array(
        [case_targets.get(c, ("", ""))[1] for c in case_str], dtype=object
    )
    missing = sorted({c for c in case_str if c not in case_targets})
    if missing:
        shown = ", ".join(missing[:10])
        more = f" and {len(missing)} cases" if len(missing) > 10 else ""
        print(f"[target] Value NPZ has no {source_label} for cases: {shown}{more}")
    return target_ids, target_names


def _collect_case_label(
    case_labels: Dict[str, Tuple[str, str]],
    case_id: str,
    label: Tuple[str, str],
    value_type: str,
    label_kind: str,
    path: Path,
) -> None:
    if not any(label):
        return
    known = case_labels.setdefault(case_id, label)
    if known != label:
        # Folds can explain different predicted classes if target-class is unset.
        # Keep the first reported class and warn about mismatches.
        print(
            f"[warning] case={case_id} has conflicting {value_type} NPZ {label_kind} values: "
            f"{known} vs {label}（{path.name}). Keeping the first."
        )


def load_value_records(
    value_input: Path, value_type: str
) -> Tuple[
    Dict[str, List[Tuple[np.ndarray, np.ndarray, Path]]],
    Dict[str, Tuple[str, str]],
    Dict[str, Tuple[str, str]],
]:
    """Return tile values by case and the target and predicted class mappings."""
    key_candidates = (
        ("patch_attributions",) if value_type == "ig"
        else ("attention", "attention_vis", "patch_attributions")
    )
    records: Dict[str, List[Tuple[np.ndarray, np.ndarray, Path]]] = {}
    case_targets: Dict[str, Tuple[str, str]] = {}
    case_preds: Dict[str, Tuple[str, str]] = {}
    for path in iter_value_npz(value_input):
        with np.load(path, allow_pickle=True) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
        key = next((k for k in key_candidates if k in arrays), None)
        if key is None or "coords" not in arrays:
            continue
        coords = np.asarray(arrays["coords"])
        values = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] < 2 or coords.shape[0] != values.size:
            print(f"[skip] Invalid value NPZ shape: {path}")
            continue
        case_id = infer_value_case_id(path, arrays)
        records.setdefault(case_id, []).append((coords[:, :2].astype(np.int64), values, path))
        _collect_case_label(
            case_targets, case_id, infer_value_target(arrays),
            value_type, "target_class", path,
        )
        _collect_case_label(
            case_preds, case_id, infer_value_pred(arrays),
            value_type, "pred_class", path,
        )
    if not records:
        raise RuntimeError(f"No usable {value_type} value NPZ: {value_input}")
    return records, case_targets, case_preds


def align_values_to_cluster(
    cluster_coords: np.ndarray,
    cluster_cases: np.ndarray,
    records: Dict[str, List[Tuple[np.ndarray, np.ndarray, Path]]],
) -> np.ndarray:
    values = np.full(cluster_coords.shape[0], np.nan, dtype=np.float64)
    case_str = cluster_cases.astype(str)
    for case_id in np.unique(case_str):
        idx = np.flatnonzero(case_str == case_id)
        candidates = records.get(case_id, [])
        if not candidates:
            raise KeyError(f"Value NPZ has no case={case_id}")
        matched = None
        for value_coords, value_arr, source in candidates:
            lookup = {(int(x), int(y)): float(v) for (x, y), v in zip(value_coords, value_arr)}
            row_values = np.array(
                [lookup.get((int(x), int(y)), np.nan) for x, y in cluster_coords[idx, :2]],
                dtype=np.float64,
            )
            if np.all(np.isfinite(row_values)):
                matched = row_values
                break
        if matched is None:
            source_names = ", ".join(p.name for _, _, p in candidates)
            raise ValueError(f"case={case_id} value NPZ could not be fully aligned by coordinate: {source_names}")
        values[idx] = matched
    return values


def _compute_value_polarity(
    values: np.ndarray,
    per_case_info: List[dict],
    args: argparse.Namespace,
) -> np.ndarray:
    """Compute polarity within each case so large cases do not dominate the thresholds."""
    polarity = np.full(values.shape[0], "none", dtype=object)
    for info in per_case_info:
        s, e = int(info["start_row"]), int(info["end_row"])
        if e <= s:
            continue
        polarity[s:e] = make_polarity_labels(
            values[s:e],
            mode=args.polarity_mode,
            top_k=args.top_k,
            quantile=args.quantile,
        )
    return polarity


def save_dual_value_per_patch_csv(
    out_csv: Path,
    case_ids_all: np.ndarray,
    target_ids_all: Optional[np.ndarray],
    target_names_all: Optional[np.ndarray],
    coords_all: np.ndarray,
    cluster_id: np.ndarray,
    ig_values: np.ndarray,
    ig_polarity: np.ndarray,
    attention_values: np.ndarray,
    attention_polarity: np.ndarray,
    patch_root: Optional[Path],
    ig_norm: Optional[np.ndarray] = None,
    attention_norm: Optional[np.ndarray] = None,
    ig_norm_mode: str = "none",
    attention_norm_mode: str = "none",
) -> None:
    """Write aligned IG and attention values per tile."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "target", "target_name", "coord_x", "coord_y", "cluster",
        "ig", "ig_polarity",
    ]
    if ig_norm is not None:
        fields.append(f"ig_{ig_norm_mode}")
    fields += ["attention", "attention_polarity"]
    if attention_norm is not None:
        fields.append(f"attention_{attention_norm_mode}")
    fields.append("patch_path")

    n = cluster_id.shape[0]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(n):
            case = str(case_ids_all[i])
            row = {
                "case": case,
                "target": "" if target_ids_all is None else str(target_ids_all[i]),
                "target_name": "" if target_names_all is None else str(target_names_all[i]),
                "coord_x": int(coords_all[i, 0]),
                "coord_y": int(coords_all[i, 1]),
                "cluster": int(cluster_id[i]),
                "ig": float(ig_values[i]),
                "ig_polarity": str(ig_polarity[i]),
                "attention": float(attention_values[i]),
                "attention_polarity": str(attention_polarity[i]),
            }
            if ig_norm is not None:
                row[f"ig_{ig_norm_mode}"] = float(ig_norm[i])
            if attention_norm is not None:
                row[f"attention_{attention_norm_mode}"] = float(attention_norm[i])
            if patch_root is not None:
                row["patch_path"] = str(
                    Path(patch_root) / case / _patch_filename(coords_all[i, 0], coords_all[i, 1])
                )
            else:
                row["patch_path"] = ""
            w.writerow(row)
    print(f"[saved] {out_csv}")


def save_dual_value_cluster_summary_csv(
    out_csv: Path,
    ig_stats: List[dict],
    attention_stats: List[dict],
) -> None:
    """Write IG and attention summary statistics side by side for each cluster."""
    by_ig = {int(r["cluster"]): r for r in ig_stats}
    by_attn = {int(r["cluster"]): r for r in attention_stats}
    clusters = sorted(set(by_ig) | set(by_attn))

    def add_prefixed(row: dict, prefix: str, src: dict) -> None:
        for key, value in src.items():
            if key in {"cluster", "n"}:
                continue
            row[f"{prefix}_{key.replace('attribution', 'value')}"] = value

    rows: List[dict] = []
    for c in clusters:
        ig = by_ig.get(c, {})
        attn = by_attn.get(c, {})
        row = {
            "cluster": c,
            "n": int(ig.get("n", attn.get("n", 0))),
        }
        add_prefixed(row, "ig", ig)
        add_prefixed(row, "attention", attn)
        rows.append(row)

    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}")


def build_args_from_meta(meta: dict, cli: argparse.Namespace) -> argparse.Namespace:
    saved = dict(meta.get("args", {}))

    def pick(key, default, override=None):
        if override is not None:
            return override
        return saved.get(key, default)

    ns = argparse.Namespace()
    ns.grid_rows = int(pick("grid_rows", DEFAULT_GRID_ROWS, cli.grid_rows))
    ns.grid_cols = int(pick("grid_cols", DEFAULT_GRID_COLS, cli.grid_cols))
    ns.patch_size = int(pick("patch_size", DEFAULT_PATCH_SIZE, cli.patch_size))
    ns.gap = int(pick("gap", DEFAULT_GAP, cli.gap))
    ns.seed = int(pick("seed", DEFAULT_SEED, cli.seed))

    ns.polarity_mode = str(pick("polarity_mode", "quantile", cli.polarity_mode))
    ns.top_k = int(pick("top_k", 50, cli.top_k))
    ns.quantile = float(pick("quantile", 0.2, cli.quantile))

    ns.attr_norm = str(pick("attr_norm", DEFAULT_ATTR_NORM, cli.attr_norm))
    ns.summary_clip_lo = float(pick("summary_clip_lo", DEFAULT_SUMMARY_CLIP_LO, cli.summary_clip_lo))
    ns.summary_clip_hi = float(pick("summary_clip_hi", DEFAULT_SUMMARY_CLIP_HI, cli.summary_clip_hi))
    ns.summary_robust_mode = str(
        pick("summary_robust_mode", DEFAULT_SUMMARY_ROBUST_MODE, cli.summary_robust_mode)
    )

    ns.slide_map_enabled = bool(saved.get("slide_map_enabled", DEFAULT_SLIDE_MAP_ENABLED))
    ns.slide_map_tile_size = int(
        pick("slide_map_tile_size", DEFAULT_SLIDE_MAP_TILE_SIZE, cli.slide_map_tile_size)
    )
    ns.slide_map_downsample = int(
        pick("slide_map_downsample", DEFAULT_SLIDE_MAP_DOWNSAMPLE, cli.slide_map_downsample)
    )

    ns.umap_enabled = bool(saved.get("umap_enabled", False))
    ns.umap_neighbors = int(saved.get("umap_neighbors", 0))
    ns.umap_min_dist = float(saved.get("umap_min_dist", 0.0))
    ns.umap_metric = str(saved.get("umap_metric", "euclidean"))
    ns.umap_sample = int(saved.get("umap_sample", 0))
    return ns


def rebuild_per_case_info(meta: dict, case_ids_all: np.ndarray) -> List[dict]:
    pci = meta.get("per_case_info")
    if pci:
        return [
            {
                "case": str(c.get("case", "")),
                "target": str(c.get("target", "")),
                "target_name": str(c.get("target_name", "")),
                "npz": str(c.get("npz", "")),
                "n_patches": int(c.get("n_patches", 0)),
                "start_row": int(c["start_row"]),
                "end_row": int(c["end_row"]),
            }
            for c in pci
        ]

    info: List[dict] = []
    n = len(case_ids_all)
    if n == 0:
        return info
    start = 0
    cur = str(case_ids_all[0])
    for i in range(1, n + 1):
        if i == n or str(case_ids_all[i]) != cur:
            info.append({
                "case": cur, "target": "", "target_name": "", "npz": "",
                "n_patches": i - start, "start_row": start, "end_row": i,
            })
            if i < n:
                cur = str(case_ids_all[i])
                start = i
    return info


def _map_filtered_clusters_to_full_wsi(
    coords_full: np.ndarray,
    case_ids_full: np.ndarray,
    coords_cluster: np.ndarray,
    case_ids_cluster: np.ndarray,
    cluster_id: np.ndarray,
) -> np.ndarray:
    """Map filtered labels back to full-slide coordinates by case and position when replotting uses a stricter tumor mask."""
    coords_full = np.asarray(coords_full, dtype=np.int64)
    case_ids_full = np.asarray(case_ids_full, dtype=object).reshape(-1)
    coords_cluster = np.asarray(coords_cluster, dtype=np.int64)
    case_ids_cluster = np.asarray(case_ids_cluster, dtype=object).reshape(-1)
    cluster_id = np.asarray(cluster_id, dtype=np.int64).reshape(-1)

    if not (
        coords_cluster.shape[0]
        == case_ids_cluster.shape[0]
        == cluster_id.shape[0]
    ):
        raise ValueError('Cluster coordinates, case IDs, and cluster IDs have different row counts.')
    if coords_full.shape[0] != case_ids_full.shape[0]:
        raise ValueError('Full-slide coordinates and case IDs have different row counts.')

    out = np.full(coords_full.shape[0], -1, dtype=np.int64)
    cluster_case_str = case_ids_cluster.astype(str)
    full_case_str = case_ids_full.astype(str)

    for case in np.unique(cluster_case_str):
        idx_cluster = np.flatnonzero(cluster_case_str == case)
        if idx_cluster.size == 0:
            continue
        lookup = {
            (int(x), int(y)): int(c)
            for (x, y), c in zip(
                coords_cluster[idx_cluster, :2],
                cluster_id[idx_cluster],
            )
        }
        idx_full = np.flatnonzero(full_case_str == case)
        for i in idx_full.tolist():
            key = (int(coords_full[i, 0]), int(coords_full[i, 1]))
            c = lookup.get(key)
            if c is not None:
                out[i] = c
    return out


def replot_one(
    npz_path: Path,
    json_path: Path,
    cohort_prefix: str,
    cli: argparse.Namespace,
    outputs: List[str],
) -> None:
    print(f"\n{'#' * 88}")
    print(f"[Replot] dump: {npz_path}")

    arrays, meta = load_dump(npz_path, json_path)

    cluster_id = np.asarray(arrays["cluster_id"]).astype(np.int64).reshape(-1)
    coords_all = np.asarray(arrays["coords"]).astype(np.int64)
    case_ids_all = np.asarray(arrays["case_ids"], dtype=object).reshape(-1)

    if coords_all.ndim != 2 or coords_all.shape[1] < 2:
        raise ValueError(f"Invalid dump coordinate shape: {coords_all.shape}")
    if not (cluster_id.shape[0] == coords_all.shape[0] == case_ids_all.shape[0]):
        raise ValueError('Dump cluster IDs, coordinates, and case IDs have different row counts.')

    # ------------------------------------------------------------
    # Restore full-slide coordinates from the newer dump format.
    # Use them for slide maps while retaining tumor-only analysis arrays.
    # ------------------------------------------------------------
    coords_full = None
    case_ids_full = None
    cluster_id_full_saved = None
    tumor_mask_saved = None
    per_case_info_full = None

    if "coords_full" in arrays and "case_ids_full" in arrays:
        coords_full = np.asarray(arrays["coords_full"], dtype=np.int64)
        case_ids_full = np.asarray(arrays["case_ids_full"], dtype=object).reshape(-1)

        if coords_full.ndim != 2 or coords_full.shape[1] < 2:
            raise ValueError(f"Invalid full-slide coordinate shape: {coords_full.shape}")
        if coords_full.shape[0] != case_ids_full.shape[0]:
            raise ValueError('Full-slide coordinates and full-slide case IDs have different row counts.')

        if "cluster_id_full" in arrays:
            cluster_id_full_saved = np.asarray(
                arrays["cluster_id_full"], dtype=np.int64
            ).reshape(-1)
            if cluster_id_full_saved.shape[0] != coords_full.shape[0]:
                raise ValueError(
                    'Full-slide cluster IDs and coordinates have different row counts.'
                )

        if "tumor_mask" in arrays:
            tumor_mask_saved = np.asarray(
                arrays["tumor_mask"], dtype=bool
            ).reshape(-1)
            if tumor_mask_saved.shape[0] != coords_full.shape[0]:
                raise ValueError(
                    'Tumor mask and full-slide coordinates have different row counts.'
                )

        # Allow older dumps containing full coordinates and a tumor mask only.
        if cluster_id_full_saved is None and tumor_mask_saved is not None:
            if int(tumor_mask_saved.sum()) == cluster_id.shape[0]:
                cluster_id_full_saved = np.full(
                    coords_full.shape[0], -1, dtype=np.int64
                )
                cluster_id_full_saved[tumor_mask_saved] = cluster_id

        # Map labels by coordinate when no full-slide labels are available.
        if cluster_id_full_saved is None:
            cluster_id_full_saved = _map_filtered_clusters_to_full_wsi(
                coords_full=coords_full,
                case_ids_full=case_ids_full,
                coords_cluster=coords_all,
                case_ids_cluster=case_ids_all,
                cluster_id=cluster_id,
            )

        per_case_info_full = rebuild_per_case_info({}, case_ids_full)
        print(
            f"[full WSI] Restored complete slide coordinates: "
            f"full={coords_full.shape[0]}  "
            f"clustered={int(np.sum(cluster_id_full_saved >= 0))}  "
            f"outside={int(np.sum(cluster_id_full_saved < 0))}"
        )

    # ------------------------------------------------------------
    # Read IG and attention as separate value inputs.
    # Use saved dump attributions only with --use-dump-ig.
    # ------------------------------------------------------------
    value_data: Dict[str, dict] = {}

    if cli.ig_npz is not None:
        ig_records, ig_case_targets, ig_case_preds = load_value_records(
            Path(cli.ig_npz), "ig"
        )
        ig_values = align_values_to_cluster(coords_all, case_ids_all, ig_records)
        value_data["ig"] = {
            "values": ig_values,
            "records": ig_records,
            "case_targets": ig_case_targets,
            "case_preds": ig_case_preds,
            "source": "external",
            "label": "IG attribution",
        }
    elif cli.use_dump_ig:
        if "attribution" not in arrays:
            raise KeyError('--use-dump-ig requires attribution in the dump.')
        ig_values = np.asarray(arrays["attribution"], dtype=np.float64).reshape(-1)
        if ig_values.shape[0] != cluster_id.shape[0]:
            raise ValueError('Dump attribution and cluster IDs have different row counts.')
        value_data["ig"] = {
            "values": ig_values,
            "records": None,
            "source": "dump",
            "label": "IG attribution",
        }

    if cli.attention_npz is not None:
        (
            attention_records,
            attention_case_targets,
            attention_case_preds,
        ) = load_value_records(Path(cli.attention_npz), "attention")
        attention_values = align_values_to_cluster(
            coords_all, case_ids_all, attention_records
        )
        value_data["attention"] = {
            "values": attention_values,
            "records": attention_records,
            "case_targets": attention_case_targets,
            "case_preds": attention_case_preds,
            "source": "external",
            "label": "Attention",
        }

    has_values = bool(value_data)
    if set(value_data) == {"ig", "attention"}:
        print('[mode] IG and attention')
    elif "ig" in value_data:
        print('[mode] IG only')
    elif "attention" in value_data:
        print('[mode] Attention only')
    else:
        print('[mode] Morphology only')

    target_ids_all = (
        np.asarray(arrays["target_ids"], dtype=object).reshape(-1)
        if "target_ids" in arrays else None
    )
    target_names_all = (
        np.asarray(arrays["target_names"], dtype=object).reshape(-1)
        if "target_names" in arrays else None
    )
    target_source = ""
    if target_ids_all is not None and target_names_all is not None:
        target_source = "dump"
        print('[target] Using target_ids and target_names saved in the dump.')
    else:
        # HDF5 cohort clustering can produce dumps without diagnostic labels.
        # In that case, read per-case classes from the value NPZ files.
        # Prefer target_class for attribution.
        # Use pred_class for attention-only runs.
        for label_kind, map_key in (
            ("target_class", "case_targets"),
            ("pred_class", "case_preds"),
        ):
            case_labels: Dict[str, Tuple[str, str]] = {}
            for kind in ("ig", "attention"):
                for case_id, pair in (
                    value_data.get(kind, {}).get(map_key) or {}
                ).items():
                    case_labels.setdefault(case_id, pair)
            derived = build_target_arrays_from_case_map(
                case_ids_all, case_labels, label_kind
            )
            if derived is None:
                continue
            target_ids_all, target_names_all = derived
            target_source = label_kind
            labels = sorted({name or tid for tid, name in case_labels.values()})
            print(
                f"[target] Read {label_kind} from value NPZ files for "
                f"{len(case_labels)} cases:" + ", ".join(labels)
            )
            if label_kind == "pred_class":
                print(
                    '[target] pred_class comes from the model prediction. '
                    'The resulting groups reflect predicted diagnoses.'
                )
            break

    feature_label = str(meta.get("feature_label", cohort_prefix))
    if not cohort_prefix:
        cohort_prefix = str(meta.get("cohort_prefix", f"cohort-{feature_label}"))

    args = build_args_from_meta(meta, cli)
    per_case_info = rebuild_per_case_info(meta, case_ids_all)

    # ------------------------------------------------------------
    # Apply one tumor mask to clusters and both value arrays to preserve alignment.
    # ------------------------------------------------------------
    filter_arrays: Dict[str, np.ndarray] = {
        "cluster_id": cluster_id,
        "case_ids": case_ids_all,
    }
    for kind, data in value_data.items():
        filter_arrays[f"{kind}_value"] = np.asarray(data["values"], dtype=np.float64)

    if "ig" in value_data and value_data["ig"]["source"] == "dump":
        if "polarity" in arrays:
            filter_arrays["ig_saved_polarity"] = np.asarray(
                arrays["polarity"], dtype=object
            ).reshape(-1)
        if "attribution_norm" in arrays:
            filter_arrays["ig_saved_norm"] = np.asarray(
                arrays["attribution_norm"], dtype=np.float64
            ).reshape(-1)

    if target_ids_all is not None:
        filter_arrays["target_ids"] = target_ids_all
    if target_names_all is not None:
        filter_arrays["target_names"] = target_names_all

    # Keep the pre-filter arrays for fallback handling of older dumps.
    coords_before = np.asarray(coords_all)
    cluster_before = np.asarray(cluster_id).astype(np.int64).reshape(-1)
    case_ids_before = np.asarray(case_ids_all, dtype=object).reshape(-1)
    per_case_before = list(per_case_info)

    coords_all, filtered, mask, tumor_info = apply_tumor_filter_by_case(
        coords_all,
        case_ids_all,
        arrays=filter_arrays,
        per_case_info=per_case_info,
        **tumor_filter_kwargs(cli),
    )

    cluster_id = np.asarray(filtered["cluster_id"]).astype(np.int64).reshape(-1)
    case_ids_all = np.asarray(filtered["case_ids"], dtype=object).reshape(-1)
    for kind, data in value_data.items():
        data["values"] = np.asarray(
            filtered[f"{kind}_value"], dtype=np.float64
        ).reshape(-1)

    if target_ids_all is not None:
        target_ids_all = np.asarray(filtered["target_ids"], dtype=object).reshape(-1)
    if target_names_all is not None:
        target_names_all = np.asarray(filtered["target_names"], dtype=object).reshape(-1)

    if "umap_embedding" in arrays:
        umap_embedding = np.asarray(arrays["umap_embedding"])
        umap_indices = (
            np.asarray(arrays["umap_indices"]).astype(np.int64).reshape(-1)
            if "umap_indices" in arrays else np.empty(0, dtype=np.int64)
        )
        orig_n = int(mask.shape[0])
        if umap_indices.size == 0 and umap_embedding.shape[0] == orig_n:
            arrays["umap_embedding"] = umap_embedding[mask]
        elif umap_indices.size > 0 and int(np.max(umap_indices)) < orig_n:
            keep_umap = mask[umap_indices]
            arrays["umap_embedding"] = umap_embedding[keep_umap]
            arrays["umap_indices"] = remap_indices_after_mask(umap_indices, mask)

    old_by_case = {str(c["case"]): c for c in per_case_info}
    per_case_info = []
    for span in rebuild_per_case_info({}, case_ids_all):
        old = old_by_case.get(span["case"], {})
        per_case_info.append({
            "case": span["case"],
            "target": old.get("target", ""),
            "target_name": old.get("target_name", ""),
            "npz": old.get("npz", ""),
            "n_patches": span["n_patches"],
            "start_row": span["start_row"],
            "end_row": span["end_row"],
        })

    if cli.out_dir is not None:
        cohort_dir = Path(cli.out_dir) / f"cohort_{feature_label}"
    else:
        cohort_dir = npz_path.parent
    cohort_dir.mkdir(parents=True, exist_ok=True)

    if cli.patch_root is not None:
        patch_root = Path(cli.patch_root)
    else:
        saved_patch_root = str(meta.get("patch_root", "")).strip()
        patch_root = Path(saved_patch_root) if saved_patch_root else None

    n_clusters_eff = int(meta.get("effective_n_clusters", 0))
    if cluster_id.size > 0:
        n_clusters_eff = max(n_clusters_eff, int(cluster_id.max()) + 1)
    n_clusters_eff = max(n_clusters_eff, 1)

    print(f"[Replot] feature_label={feature_label}  "
          f"cluster_method={meta.get('cluster_method', '?')}  "
          f"n_clusters={n_clusters_eff}")
    print(f"[replot] Total patches={cluster_id.shape[0]}  cases={len(per_case_info)}")
    print(f"[replot] Output directory={cohort_dir}")
    print(f"[Replot] patch_root={patch_root}")
    print(f"{'#' * 88}")

    if outputs:
        want = set(outputs)
    else:
        want = set(ALL_OUTPUTS if has_values else MORPHOLOGY_ONLY_OUTPUTS)

    if not has_values:
        if "per_patch" in want:
            save_morphology_per_patch_csv(
                cohort_dir / f"{cohort_prefix}-morphology-per_patch.csv",
                case_ids_all, coords_all, cluster_id, patch_root,
            )
        skipped = sorted(x for x in want if x in VALUE_REQUIRED_OUTPUTS)
        if skipped:
            print('[skip] The following outputs require IG or attention:' + ", ".join(skipped))
        want -= VALUE_REQUIRED_OUTPUTS

    # ------------------------------------------------------------
    # Compute polarity, normalization, and statistics for each value type.
    # ------------------------------------------------------------
    for kind, data in value_data.items():
        values = np.asarray(data["values"], dtype=np.float64)
        source = str(data["source"])

        if kind == "ig":
            norm_mode = str(args.attr_norm).lower()
        else:
            norm_mode = str(cli.attention_norm or DEFAULT_ATTENTION_NORM).lower()

        if (
            kind == "ig"
            and source == "dump"
            and not cli.recompute_polarity
            and "ig_saved_polarity" in filtered
        ):
            polarity = np.asarray(
                filtered["ig_saved_polarity"], dtype=object
            ).reshape(-1)
            print('[IG polarity] Using the labels saved in the dump.')
        else:
            polarity = _compute_value_polarity(values, per_case_info, args)
            print(
                f"[{kind.upper()} polarity] Recomputed using mode={args.polarity_mode} "
                'within each case.'
            )

        values_norm = None
        if norm_mode != "none":
            saved_norm_mode = str(meta.get("attr_norm", {}).get("mode", "")).lower()
            if (
                kind == "ig"
                and source == "dump"
                and not cli.recompute_polarity
                and "ig_saved_norm" in filtered
                and saved_norm_mode == norm_mode
            ):
                values_norm = np.asarray(
                    filtered["ig_saved_norm"], dtype=np.float64
                ).reshape(-1)
                print(f"[IG normalization] Using saved values: {norm_mode}")
            else:
                values_norm = normalize_attrs_per_case(
                    attrs_all=values,
                    per_case_info=per_case_info,
                    mode=norm_mode,
                )
                print(f"[{kind.upper()} normalization] mode={norm_mode} (per-case)")

        stats = compute_cluster_summary(
            values,
            cluster_id,
            polarity,
            n_clusters_eff,
            attrs_norm=values_norm,
            robust_mode=args.summary_robust_mode,
            clip_lo=args.summary_clip_lo,
            clip_hi=args.summary_clip_hi,
        )

        cases, count_total, count_pos, count_neg = compute_case_cluster_polarity_table(
            case_ids_all, cluster_id, polarity, n_clusters_eff
        )

        data["polarity"] = polarity
        data["norm"] = values_norm
        data["norm_mode"] = norm_mode
        data["stats"] = stats
        data["cases"] = cases
        data["count_total"] = count_total
        data["count_pos"] = count_pos
        data["count_neg"] = count_neg

        value_prefix = f"{cohort_prefix}-{kind}"
        value_label = str(data["label"])

        if "per_patch" in want:
            save_per_patch_csv(
                cohort_dir / f"{value_prefix}-per_patch.csv",
                case_ids_all=case_ids_all,
                target_ids_all=target_ids_all,
                target_names_all=target_names_all,
                coords_all=coords_all,
                attrs_all=values,
                cluster_id=cluster_id,
                polarity=polarity,
                patch_root=patch_root,
                attrs_norm=values_norm,
                attr_norm_mode=norm_mode,
            )

        if "cluster_summary" in want:
            save_cluster_summary_csv(
                cohort_dir / f"{value_prefix}-cluster_summary.csv",
                stats,
            )

        if (
            "target_summary" in want
            and target_ids_all is not None
            and target_names_all is not None
        ):
            target_stats = compute_target_cluster_summary(
                attrs_all=values,
                cluster_id=cluster_id,
                polarity=polarity,
                target_ids_all=target_ids_all,
                target_names_all=target_names_all,
                n_clusters=n_clusters_eff,
                attrs_norm=values_norm if norm_mode != "none" else None,
                robust_mode=args.summary_robust_mode,
                clip_lo=args.summary_clip_lo,
                clip_hi=args.summary_clip_hi,
            )
            save_cluster_summary_csv(
                cohort_dir / f"{value_prefix}-target_cluster_summary.csv",
                target_stats,
            )
            save_target_cluster_rank_order_csv(
                cohort_dir / f"{value_prefix}-target_cluster_rank_order.csv",
                target_stats,
                target_source=target_source,
            )
        elif "target_summary" in want:
            print(
                f"[skip] {kind} target summary has no target_ids or target_names in the dump, "
                'and the value NPZ has no target_class or pred_class for grouping.'
            )

        if "case_polarity" in want:
            save_case_cluster_polarity_csv(
                cohort_dir / f"{value_prefix}-case_cluster_polarity.csv",
                cases, count_total, count_pos, count_neg,
            )

        if "box" in want:
            box_values = values_norm if norm_mode != "none" else values
            box_ylabel = (
                f"{value_label} (per-case {norm_mode})"
                if norm_mode != "none" else value_label
            )
            valid_stats = [s for s in stats if s["n"] > 0]
            if valid_stats:
                norm_active = (
                    norm_mode != "none"
                    and "mean_attribution_norm" in valid_stats[0]
                )
                sort_key = (
                    "mean_attribution_norm" if norm_active
                    else "mean_attribution"
                )
                pos_cluster = int(max(valid_stats, key=lambda s: s[sort_key])["cluster"])
                neg_cluster = int(min(valid_stats, key=lambda s: s[sort_key])["cluster"])
                sort_label = (
                    f"mean_{kind}_{norm_mode}" if norm_active
                    else f"mean_{kind}"
                )
            else:
                pos_cluster = neg_cluster = None
                sort_label = None

            save_cluster_boxplot(
                cohort_dir / f"{value_prefix}-cluster_box.png",
                attrs_all=box_values,
                cluster_id=cluster_id,
                n_clusters=n_clusters_eff,
                title=(
                    f"{cohort_prefix}  {value_label} per global cluster "
                    f"(all cases pooled, K={n_clusters_eff}, norm={norm_mode})"
                ),
                ylabel=box_ylabel,
                highlight_positive=pos_cluster,
                highlight_negative=neg_cluster,
                sort_key_label=sort_label,
            )

        if "heatmap" in want:
            save_case_cluster_heatmap(
                cohort_dir / f"{value_prefix}-case_cluster_heatmap.png",
                cases=cases,
                count_total=count_total,
                count_pos=count_pos,
                count_neg=count_neg,
                title=f"{cohort_prefix}  {value_label} case × cluster polarity preference",
            )

        if "grids" in want:
            if patch_root is None:
                print(f"[skip] {kind} global cluster grids require patch_root.")
            else:
                save_global_cluster_grids(
                    grids_dir=cohort_dir / "global_clusters_grid" / kind,
                    cohort_prefix=value_prefix,
                    cluster_stats=stats,
                    cluster_id=cluster_id,
                    case_ids_all=case_ids_all,
                    coords_all=coords_all,
                    attrs_all=values,
                    polarity=polarity,
                    per_case_info=per_case_info,
                    patch_root=patch_root,
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                    patch_size=args.patch_size,
                    gap=args.gap,
                    seed=args.seed,
                    attr_norm_mode=norm_mode,
                )

        if "polarity_grids" in want:
            if patch_root is None:
                print(f"[skip] {kind} polarity grids require patch_root.")
            else:
                save_polarity_representative_grids(
                    out_dir=cohort_dir,
                    cohort_prefix=value_prefix,
                    cluster_stats=stats,
                    cluster_id=cluster_id,
                    case_ids_all=case_ids_all,
                    coords_all=coords_all,
                    patch_root=patch_root,
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                    patch_size=args.patch_size,
                    gap=args.gap,
                    seed=args.seed,
                    attr_norm_mode=norm_mode,
                )

    # ------------------------------------------------------------
    # Write joint per-patch and per-cluster tables in dual-value mode.
    # ------------------------------------------------------------
    if "ig" in value_data and "attention" in value_data:
        ig = value_data["ig"]
        attn = value_data["attention"]

        if "per_patch" in want:
            save_dual_value_per_patch_csv(
                cohort_dir / f"{cohort_prefix}-ig_attention-per_patch.csv",
                case_ids_all=case_ids_all,
                target_ids_all=target_ids_all,
                target_names_all=target_names_all,
                coords_all=coords_all,
                cluster_id=cluster_id,
                ig_values=ig["values"],
                ig_polarity=ig["polarity"],
                attention_values=attn["values"],
                attention_polarity=attn["polarity"],
                patch_root=patch_root,
                ig_norm=ig["norm"],
                attention_norm=attn["norm"],
                ig_norm_mode=ig["norm_mode"],
                attention_norm_mode=attn["norm_mode"],
            )

        if "cluster_summary" in want:
            save_dual_value_cluster_summary_csv(
                cohort_dir / f"{cohort_prefix}-ig_attention-cluster_summary.csv",
                ig_stats=ig["stats"],
                attention_stats=attn["stats"],
            )

    # ------------------------------------------------------------
    # Write morphology outputs once for both value types.
    # ------------------------------------------------------------
    if "umap" in want:
        if "umap_embedding" in arrays:
            umap_embedding = np.asarray(arrays["umap_embedding"]).astype(np.float32)
            umap_indices = None
            if "umap_indices" in arrays:
                ui = np.asarray(arrays["umap_indices"]).astype(np.int64).reshape(-1)
                umap_indices = ui if ui.size > 0 else None
            save_umap_scatter(
                cohort_dir / f"{cohort_prefix}-umap_scatter.png",
                embedding=umap_embedding,
                cluster_id=cluster_id,
                n_clusters=n_clusters_eff,
                title=f"{cohort_prefix}  UMAP 2D (cluster color, K={n_clusters_eff})",
                indices=umap_indices,
                seed=args.seed,
            )
            save_umap_scatter(
                cohort_dir / f"{cohort_prefix}-umap_scatter_legend.png",
                embedding=umap_embedding,
                cluster_id=cluster_id,
                n_clusters=n_clusters_eff,
                title=f"{cohort_prefix}  UMAP 2D (cluster color, K={n_clusters_eff})",
                indices=umap_indices,
                seed=args.seed,
                bottom_legend=True,
            )
        else:
            print('[skip] The dump has no UMAP embedding to replot.')

    if (not has_values) and "grids" in want:
        if patch_root is None:
            print('[skip] Morphology global cluster grids require patch_root.')
        else:
            empty_values = np.zeros(cluster_id.shape[0], dtype=np.float64)
            empty_polarity = np.full(cluster_id.shape[0], "none", dtype=object)
            morphology_stats = compute_cluster_summary(
                empty_values,
                cluster_id,
                empty_polarity,
                n_clusters_eff,
                attrs_norm=None,
                robust_mode="none",
            )
            save_global_cluster_grids(
                grids_dir=cohort_dir / "global_clusters_grid",
                cohort_prefix=cohort_prefix,
                cluster_stats=morphology_stats,
                cluster_id=cluster_id,
                case_ids_all=case_ids_all,
                coords_all=coords_all,
                attrs_all=empty_values,
                polarity=empty_polarity,
                per_case_info=per_case_info,
                patch_root=patch_root,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                patch_size=args.patch_size,
                gap=args.gap,
                seed=args.seed,
                attr_norm_mode="none",
            )

    if "slide_map" in want:
        display_mode = str(
            getattr(cli, "tumor_display_mode", DEFAULT_TUMOR_DISPLAY_MODE)
        )

        # --------------------------------------------------------
        # Use full-slide arrays for zero_outside with tumor filtering.
        # Other modes use the filtered data for slide maps.
        # --------------------------------------------------------
        if (
            bool(tumor_info.get("enabled"))
            and display_mode == "zero_outside"
            and coords_full is not None
            and case_ids_full is not None
            and cluster_id_full_saved is not None
            and per_case_info_full is not None
        ):
            # With the same annotation, all previously retained patches pass the filter.
            # Reuse the full-slide labels saved in the dump.
            mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
            if (
                mask_arr.shape[0] == cluster_before.shape[0]
                and bool(np.all(mask_arr))
            ):
                slide_cluster = cluster_id_full_saved
            else:
                # Replotting may use a stricter tumor annotation.
                # Map the surviving labels to the full slide and leave other tiles gray.
                slide_cluster = _map_filtered_clusters_to_full_wsi(
                    coords_full=coords_full,
                    case_ids_full=case_ids_full,
                    coords_cluster=coords_all,
                    case_ids_cluster=case_ids_all,
                    cluster_id=cluster_id,
                )

            slide_coords = coords_full
            slide_cases = case_ids_full
            slide_spans = per_case_info_full

            print(
                '[region] The cluster map uses the full-slide outline from the dump. '
                f"Non-tumor or unclustered patches={int(np.sum(slide_cluster < 0))} shown in gray."
            )

        # --------------------------------------------------------
        # Fall back to the available coordinates for older dumps.
        # --------------------------------------------------------
        elif (
            bool(tumor_info.get("enabled"))
            and display_mode == "zero_outside"
        ):
            slide_cluster = cluster_before.copy()
            slide_cluster[~mask] = -1
            slide_coords = coords_before
            slide_cases = case_ids_before
            slide_spans = per_case_before
            print(
                '[region] The cluster map uses the available slide outline. '
                f"Non-tumor patches={int((~mask).sum())} shown in gray."
            )
        else:
            slide_cluster = cluster_id
            slide_coords = coords_all
            slide_cases = case_ids_all
            slide_spans = per_case_info

        save_cohort_case_cluster_maps(
            out_dir=cohort_dir / "case_cluster_slide_map",
            cohort_prefix=cohort_prefix,
            cluster_id=slide_cluster,
            coords_all=slide_coords,
            case_ids_all=slide_cases,
            per_case_info=slide_spans,
            n_clusters=n_clusters_eff,
            tile_size=int(args.slide_map_tile_size),
            downsample=int(args.slide_map_downsample),
        )

    if cli.recompute_polarity and not has_values:
        print('[skip] Polarity recomputation requires IG or attention values.')

    print(f"[replot complete] {cohort_prefix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Replot existing cohort clusters with optional IG and attention values.'
    )

    p.add_argument(
        "--ig-npz",
        type=optional_path,
        default=DEFAULT_IG_NPZ,
        help='IG NPZ file or directory containing coords and patch_attributions. '
             'Use none to disable IG.',
    )
    p.add_argument(
        "--attention-npz",
        type=optional_path,
        default=DEFAULT_ATTENTION_NPZ,
        help='Attention NPZ file or directory containing coords and attention values. '
             'Use none to disable attention.',
    )
    p.add_argument(
        "--use-dump-ig",
        action="store_true",
        default=DEFAULT_USE_DUMP_IG,
        help='Use saved dump attributions when --ig-npz is omitted',
    )

    p.add_argument(
        "--dump",
        type=Path,
        default=DEFAULT_DUMP_INPUT,
        help='Cohort directory, cluster-result NPZ or JSON, or parent of cohort directories',
    )
    p.add_argument(
        "--patch-root",
        type=Path,
        default=DEFAULT_PATCH_ROOT,
        help='Override the patch image root saved in the dump',
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help='Output root. Defaults to the dump directory.',
    )
    p.add_argument(
        "--outputs",
        nargs="*",
        default=DEFAULT_OUTPUTS,
        choices=ALL_OUTPUTS,
        help='Choose outputs to generate. Defaults to every output available for the inputs.',
    )

    p.add_argument("--grid-rows", type=int, default=DEFAULT_CLI_GRID_ROWS)
    p.add_argument("--grid-cols", type=int, default=DEFAULT_CLI_GRID_COLS)
    p.add_argument("--patch-size", type=int, default=DEFAULT_CLI_PATCH_SIZE)
    p.add_argument("--gap", type=int, default=DEFAULT_CLI_GAP)
    p.add_argument("--seed", type=int, default=DEFAULT_CLI_SEED)

    p.add_argument(
        "--recompute-polarity",
        action="store_true",
        default=DEFAULT_RECOMPUTE_POLARITY,
        help='Recompute polarity using the selected settings',
    )
    p.add_argument(
        "--polarity-mode",
        choices=["top_k", "sign", "quantile"],
        default=DEFAULT_CLI_POLARITY_MODE,
    )
    p.add_argument("--top-k", type=int, default=DEFAULT_CLI_TOP_K)
    p.add_argument("--quantile", type=float, default=DEFAULT_CLI_QUANTILE)

    p.add_argument(
        "--attr-norm",
        choices=list(ATTR_NORM_MODES),
        default=DEFAULT_CLI_ATTR_NORM,
        help='Per-case IG normalization. Defaults to the saved or module settings.',
    )
    p.add_argument(
        "--attention-norm",
        choices=list(ATTR_NORM_MODES),
        default=DEFAULT_ATTENTION_NORM,
        help='Per-case attention normalization. Defaults to none.',
    )

    p.add_argument("--summary-clip-lo", type=float, default=DEFAULT_CLI_SUMMARY_CLIP_LO)
    p.add_argument("--summary-clip-hi", type=float, default=DEFAULT_CLI_SUMMARY_CLIP_HI)
    p.add_argument(
        "--summary-robust-mode",
        choices=list(SUMMARY_ROBUST_MODES),
        default=DEFAULT_CLI_SUMMARY_ROBUST_MODE,
    )

    p.add_argument(
        "--slide-map-tile-size",
        type=int,
        default=DEFAULT_CLI_SLIDE_MAP_TILE_SIZE,
    )
    p.add_argument(
        "--slide-map-downsample",
        type=int,
        default=DEFAULT_CLI_SLIDE_MAP_DOWNSAMPLE,
    )
    add_tumor_region_arguments(
        p,
        single_csv=False,
        annotation_dir_default=DEFAULT_ANNOTATION_DIR,
        tumor_filter_default=DEFAULT_TUMOR_FILTER,
        display_mode_default=DEFAULT_TUMOR_DISPLAY_MODE,
    )
    p.set_defaults(tumor_filter=DEFAULT_TUMOR_FILTER)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    pairs = find_dump_pairs(Path(cli.dump))
    if not pairs:
        raise FileNotFoundError(
            f"No *-cluster_result.npz files found under {cli.dump}."
            f"Generate a cohort clustering dump first."
        )

    print(f"[config] Found {len(pairs)} cohort dumps:")
    for npz_path, _json_path, prefix in pairs:
        print(f"   · {prefix:<24} <- {npz_path}")

    for npz_path, json_path, prefix in pairs:
        try:
            replot_one(npz_path, json_path, prefix, cli, cli.outputs)
        except Exception as e:
            print(f"[error] Dump {npz_path} could not be replotted: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 88}\n[complete] Processed {len(pairs)} cohort dumps")


if __name__ == "__main__":
    main()
