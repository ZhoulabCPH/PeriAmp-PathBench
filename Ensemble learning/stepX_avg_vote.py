"""
Evaluate mean and voting ensembles of selected backbones.
Calculate softmax from each input CSV's logit columns and average probabilities across folds. Combine MIL models within each backbone before combining backbones. Save OOF and external predictions under 03_ablation_ensemble and metrics and plots under 04_metrics/ablation_ensemble. Accept raw MIL CSVs or Step 0 probability CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import metrics as metrics_mod
from hybridb_common import (
    ensure_dir,
    is_external,
    is_internal,
    label_to_index,
    normalize_dataset_type,
    safe_name,
    save_metrics_bundle,
)
from softmax_probs import CLASS_ORDER, PROB_COLS, with_softmax_probs


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}][Ablation] {msg}", flush=True)


def _discover_csvs(input_dir: Path) -> List[Path]:
    csvs = sorted(input_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under input_dir: {input_dir}")
    return csvs


def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def _read_csv_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Could not read CSV: {path}")


def _progress_bar(i: int, n: int, width: int = 28) -> str:
    if n <= 0:
        return "[????????????????????????????]"
    i = max(0, min(i, n))
    filled = int(round(width * (i / n)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _parse_big_model_and_mil(path: Path) -> Tuple[str, str]:
    """Return (backbone, MIL) from a Step 0 filename <MIL>__<Backbone>.csv or a raw prediction path <MIL>/<Backbone>.csv."""
    stem = path.stem
    if "__" in stem:
        mil, backbone = stem.split("__", 1)
        return safe_name(backbone), safe_name(mil)
    # Use the parent directory as the MIL name and the filename stem as the backbone.
    return safe_name(stem), safe_name(path.parent.name)


def _load_one_model_csv(path: Path, model_id: str) -> pd.DataFrame:
    df = with_softmax_probs(pd.read_csv(path))
    for col in ["file_id", "fold", "dataset_type"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column {col}")

    df["dataset_type"] = df["dataset_type"].apply(normalize_dataset_type)
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype(object)
    # Set external folds to "all" to align models as in load_probability_csv.
    df.loc[df["dataset_type"].apply(is_external), "fold"] = "all"

    prob_cols = PROB_COLS

    key_cols = ["file_id", "fold", "dataset_type"]
    has_label = "true_label" in df.columns
    cols = key_cols + (["true_label"] if has_label else []) + prob_cols
    sub = df[cols].copy()

    # Deduplicate keys to prevent a many-to-many join.
    if sub.duplicated(subset=key_cols).any():
        agg_probs = sub.groupby(key_cols, as_index=False)[prob_cols].mean()
        if has_label:
            def _resolve_label(s: pd.Series):
                non_null = s.dropna().astype(str)
                uniq = non_null.unique().tolist()
                if len(uniq) == 0:
                    return np.nan
                if len(uniq) > 1:
                    raise ValueError(f"{path} has multiple true_label values for the same (file_id, fold, dataset_type): {uniq}")
                return uniq[0]

            lbl = sub.groupby(key_cols, as_index=False)["true_label"].agg(_resolve_label)
            sub = pd.merge(agg_probs, lbl, on=key_cols, how="left")
            sub = sub[key_cols + ["true_label"] + prob_cols]
        else:
            sub = agg_probs

    # Name probability columns <model_id>__<CLASS>.
    rename_map = {src: f"{model_id}__{cls}" for src, cls in zip(prob_cols, CLASS_ORDER)}
    sub = sub.rename(columns=rename_map)
    return sub


def _merge_models(csv_paths: List[Path]) -> Tuple[pd.DataFrame, List[str]]:
    merged: Optional[pd.DataFrame] = None
    model_ids: List[str] = []
    for p in csv_paths:
        # Build a stable, unique model identifier.
        # Step 0 filenames already use <MIL>__<Backbone>.
        # Raw paths use <MIL>/<Backbone>.csv and can have repeated stems.
        if "__" in p.stem:
            model_id = safe_name(p.stem)
        else:
            model_id = safe_name(f"{p.parent.name}__{p.stem}")
        # Append a sequence number when identifiers still collide.
        if model_id in set(model_ids):
            base = model_id
            k = 2
            while f"{base}__dup{k}" in set(model_ids):
                k += 1
            model_id = f"{base}__dup{k}"
        df_m = _load_one_model_csv(p, model_id=model_id)
        model_ids.append(model_id)
        if merged is None:
            merged = df_m
        else:
            key_cols = ["file_id", "fold", "dataset_type"]
            if "true_label" in merged.columns and "true_label" in df_m.columns:
                key_cols = key_cols + ["true_label"]
            merged = pd.merge(merged, df_m, on=key_cols, how="outer")
    if merged is None:
        raise RuntimeError('The merge is empty because no model CSVs were loaded.')

    return merged, model_ids


def _get_model_prob_matrix(merged: pd.DataFrame, model_ids: List[str]) -> np.ndarray:
    """Return probabilities with shape (n_samples, n_models, n_classes)."""
    mats = []
    for mid in model_ids:
        cols = [f"{mid}__{cls}" for cls in CLASS_ORDER]
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            raise ValueError(f"The merged table is missing a column for model {mid}）：{missing}")
        mats.append(merged[cols].to_numpy(dtype=np.float64))
    return np.stack(mats, axis=1)


def _renorm(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    return p / s


def _ensemble_mean(p_models: np.ndarray) -> np.ndarray:
    # p_models: (n, m, c)
    p = np.mean(p_models, axis=1)
    return _renorm(p)


def _ensemble_vote(p_models: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-class vote fractions and predicted class indices. Break ties by mean model probability, then by CLASS_ORDER."""
    n, m, c = p_models.shape
    picks = np.argmax(p_models, axis=2)  # (n, m)
    counts = np.zeros((n, c), dtype=np.int64)
    for k in range(c):
        counts[:, k] = np.sum(picks == k, axis=1)

    tied = counts == counts.max(axis=1, keepdims=True)
    mean_probs = np.mean(p_models, axis=1)
    pred = np.argmax(np.where(tied, mean_probs, -np.inf), axis=1)

    vote_frac = counts.astype(np.float64) / float(max(m, 1))
    vote_frac = _renorm(vote_frac)
    return vote_frac, pred


def _aggregate_big_models_from_mils(
    csv_paths: List[Path],
    *,
    do_intersection: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Join predictions by file ID, fold, and dataset. Aggregate MIL predictions within each backbone into mean probability columns and a voting prediction index to reduce memory use during combination search."""
    if not csv_paths:
        raise ValueError('No CSV paths were provided.')

    # big_model -> list[(mil_id, path)]
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for p in csv_paths:
        bm, mil = _parse_big_model_and_mil(p)
        groups.setdefault(bm, []).append((mil, p))

    big_models = sorted(groups.keys())
    _log(f"Found backbones={len(big_models)}, total MIL CSVs={len(csv_paths)}")
    if not big_models:
        raise RuntimeError('Could not group any CSVs by backbone.')

    merged_base: Optional[pd.DataFrame] = None

    for bm in big_models:
        mil_items = sorted(groups[bm], key=lambda x: x[0])
        mil_ids = [m for m, _ in mil_items]
        if len(mil_ids) == 0:
            continue

        # Read each MIL model and prefix columns with its backbone and MIL names.
        mil_dfs: List[pd.DataFrame] = []
        for mil, path in mil_items:
            mil_model_id = safe_name(f"{bm}__{mil}")
            df_m = _load_one_model_csv(path, model_id=mil_model_id)
            mil_dfs.append(df_m)

        # Join MIL predictions within the current backbone.
        df_bm: Optional[pd.DataFrame] = None
        for df_m in mil_dfs:
            if df_bm is None:
                df_bm = df_m
            else:
                key_cols = ["file_id", "fold", "dataset_type"]
                if "true_label" in df_bm.columns and "true_label" in df_m.columns:
                    key_cols = key_cols + ["true_label"]
                df_bm = pd.merge(df_bm, df_m, on=key_cols, how="outer")
        if df_bm is None:
            raise RuntimeError(f"{bm} produced an empty aggregate.")

        # Require all MIL predictions for each included sample.
        prob_cols_all: List[str] = []
        for mil in mil_ids:
            mil_model_id = safe_name(f"{bm}__{mil}")
            prob_cols_all.extend([f"{mil_model_id}__{cls}" for cls in CLASS_ORDER])
        if do_intersection:
            df_bm = df_bm.dropna(subset=prob_cols_all).copy()

        # Output the sample key, mean probabilities, and vote.
        out = df_bm[["file_id", "fold", "dataset_type"]].copy()
        if "true_label" in df_bm.columns:
            out["true_label"] = df_bm["true_label"]

        # Average probabilities across MIL models.
        for cls in CLASS_ORDER:
            cols = [f"{safe_name(f'{bm}__{mil}')}__{cls}" for mil in mil_ids]
            out[f"{bm}__{cls}"] = df_bm[cols].mean(axis=1).astype(np.float64)

        # Vote across MIL model labels.
        mil_prob_mats = []
        for mil in mil_ids:
            cols = [f"{safe_name(f'{bm}__{mil}')}__{cls}" for cls in CLASS_ORDER]
            mil_prob_mats.append(df_bm[cols].to_numpy(dtype=np.float64))
        p_mils = np.stack(mil_prob_mats, axis=1)  # (n, n_mil, c)
        _, out[f"{bm}__pick"] = _ensemble_vote(p_mils)

        # Use the shared "all" label for external folds.
        out["dataset_type"] = out["dataset_type"].apply(normalize_dataset_type)
        out.loc[out["dataset_type"].apply(is_external), "fold"] = "all"

        # Join the backbone into merged_base.
        if merged_base is None:
            merged_base = out
        else:
            key_cols = ["file_id", "fold", "dataset_type"]
            if "true_label" in merged_base.columns and "true_label" in out.columns:
                key_cols = key_cols + ["true_label"]
            merged_base = pd.merge(merged_base, out, on=key_cols, how="outer")

    if merged_base is None:
        raise RuntimeError('The backbone aggregate is empty.')


    return merged_base, big_models


def _parse_list_arg(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    # Accept plus-separated and comma-separated combination strings.
    for sep in [";", ",", "+"]:
        x = x.replace(sep, ",")
    return [p.strip() for p in x.split(",") if p.strip()]


def _summary_row_to_models(row: Dict) -> List[str]:
    models = str(row.get("models", "") or "").strip()
    return _parse_list_arg(models)


def _resolve_combo_for_explain(
    *,
    combo_arg: str,
    summary_path: Path,
    big_models: Sequence[str],
) -> Tuple[List[str], str, str, Optional[Dict]]:
    """Resolve a run_id, tag, or comma- or plus-separated model list. Default to the top-ranked entry when summary_human.csv exists."""
    big_model_set = set(big_models)
    row: Optional[Dict] = None
    combo_arg = (combo_arg or "").strip()

    if summary_path.exists():
        df = _read_csv_fallback(summary_path)
        if len(df) > 0:
            if combo_arg:
                masks = []
                for col in ["run_id", "tag"]:
                    if col in df.columns:
                        masks.append(df[col].astype(str) == combo_arg)
                if masks:
                    mask = masks[0]
                    for m in masks[1:]:
                        mask = mask | m
                    hit = df[mask].copy()
                    if len(hit) > 0:
                        row = hit.iloc[0].to_dict()
            else:
                if "rank" in df.columns:
                    df2 = df.copy()
                    df2["rank"] = pd.to_numeric(df2["rank"], errors="coerce")
                    df2 = df2.sort_values("rank", na_position="last")
                    row = df2.iloc[0].to_dict()
                else:
                    row = df.iloc[0].to_dict()

    if row is not None:
        combo_models = _summary_row_to_models(row)
        if not combo_models:
            raise ValueError(f"The matched summary row has no models: {combo_arg}")
        missing = [m for m in combo_models if m not in big_model_set]
        if missing:
            raise ValueError(f"Models in the matched summary row are missing from the inputs: {missing}")
        combo_key = "+".join(combo_models)
        tag = str(row.get("tag") or f"combo_k{len(combo_models)}_{_short_hash(combo_key)}")
        run_id = str(row.get("run_id") or _short_hash(combo_key))
        return combo_models, tag, run_id, row

    combo_models = _parse_list_arg(combo_arg)
    if not combo_models:
        raise ValueError(
            'Could not select a combination from summary_human.csv. '
            'Provide a run ID, tag, or list of models such as A+B+C.'
        )
    missing = [m for m in combo_models if m not in big_model_set]
    if missing:
        raise ValueError(f"Unknown backbones in --shap_combo: {missing}\nAvailable backbones: {list(big_models)}")
    combo_key = "+".join(combo_models)
    run_id = _short_hash(combo_key)
    tag = f"combo_k{len(combo_models)}_{run_id}"
    return combo_models, tag, run_id, None


def _load_combo_mil_detail(
    csv_paths: List[Path],
    combo_models: Sequence[str],
    *,
    do_intersection: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Reload per-MIL probabilities for the selected backbone combination with columns <backbone>__<mil>__<class>."""
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for p in csv_paths:
        bm, mil = _parse_big_model_and_mil(p)
        groups.setdefault(bm, []).append((mil, p))

    merged: Optional[pd.DataFrame] = None
    bm_to_mils: Dict[str, List[str]] = {}

    for bm in combo_models:
        if bm not in groups:
            raise ValueError(f"No CSVs found for backbone {bm}.")
        mil_items = sorted(groups[bm], key=lambda x: x[0])
        bm_to_mils[bm] = [mil for mil, _ in mil_items]
        for mil, path in mil_items:
            mil_model_id = safe_name(f"{bm}__{mil}")
            df_m = _load_one_model_csv(path, model_id=mil_model_id)
            if merged is None:
                merged = df_m
            else:
                key_cols = ["file_id", "fold", "dataset_type"]
                if "true_label" in merged.columns and "true_label" in df_m.columns:
                    key_cols = key_cols + ["true_label"]
                merged = pd.merge(merged, df_m, on=key_cols, how="outer")

    if merged is None:
        raise RuntimeError('No predictions could be joined for this combination.')

    prob_cols: List[str] = []
    for bm, mils in bm_to_mils.items():
        for mil in mils:
            mid = safe_name(f"{bm}__{mil}")
            prob_cols.extend([f"{mid}__{cls}" for cls in CLASS_ORDER])
    if do_intersection:
        before = len(merged)
        merged = merged.dropna(subset=prob_cols).copy()
        after = len(merged)
        if after == 0:
            raise ValueError('No samples have complete predictions for this combination.')
        if after != before:
            _log(f"[SHAP] combo intersection: {before} -> {after}")

    merged["dataset_type"] = merged["dataset_type"].apply(normalize_dataset_type)
    merged.loc[merged["dataset_type"].apply(is_external), "fold"] = "all"
    merged = merged.reset_index(drop=True)

    return merged, bm_to_mils


def _build_combo_scores_from_mil_detail(
    df: pd.DataFrame,
    combo_models: Sequence[str],
    bm_to_mils: Dict[str, List[str]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    bm_probs: Dict[str, np.ndarray] = {}
    bm_picks: Dict[str, np.ndarray] = {}

    for bm in combo_models:
        mil_mats = []
        for mil in bm_to_mils[bm]:
            mid = safe_name(f"{bm}__{mil}")
            cols = [f"{mid}__{cls}" for cls in CLASS_ORDER]
            mil_mats.append(df[cols].to_numpy(dtype=np.float64))
        p_mils = np.stack(mil_mats, axis=1)  # (n, n_mil, c)
        bm_probs[bm] = _renorm(np.mean(p_mils, axis=1))

        _, bm_picks[bm] = _ensemble_vote(p_mils)

    p_models = np.stack([bm_probs[bm] for bm in combo_models], axis=1)
    p_mean = _renorm(np.mean(p_models, axis=1))

    picks = np.stack([bm_picks[bm] for bm in combo_models], axis=1)
    counts = np.zeros((len(df), len(CLASS_ORDER)), dtype=np.int64)
    for cls_idx in range(len(CLASS_ORDER)):
        counts[:, cls_idx] = np.sum(picks == cls_idx, axis=1)
    p_vote = _renorm(counts.astype(np.float64) / float(max(len(combo_models), 1)))
    return p_mean, p_vote, bm_probs, bm_picks


def _select_explain_dataset(df: pd.DataFrame, y_score: np.ndarray, dataset: str) -> Tuple[pd.DataFrame, np.ndarray]:
    ds = safe_name((dataset or "INTERNAL_OOF").strip().upper())
    if ds == "INTERNAL_OOF":
        mask = df["dataset_type"].apply(is_internal).to_numpy()
    elif ds == "EXTERNAL_MERGE":
        mask = df["dataset_type"].apply(is_external).to_numpy()
    else:
        mask = df["dataset_type"].astype(str).apply(lambda x: safe_name(x.upper()) == ds).to_numpy()

    if not np.any(mask):
        raise ValueError(f"The selected dataset is empty: {dataset}")
    out = df.loc[mask].copy()
    out["__source_index"] = np.flatnonzero(mask)
    if ds != "INTERNAL_OOF":
        out = out.drop_duplicates(subset=["file_id", "dataset_type"]).copy()
    src_idx = out["__source_index"].to_numpy(dtype=np.int64)
    return out.reset_index(drop=True), y_score[src_idx]


def _target_class_index(df: pd.DataFrame, y_score: np.ndarray, class_arg: str) -> np.ndarray:
    class_arg = (class_arg or "pred").strip()
    if class_arg.lower() == "pred":
        return np.argmax(y_score, axis=1).astype(np.int64)
    if class_arg.lower() == "true":
        if "true_label" not in df.columns:
            raise ValueError('--shap_class=true requires a true_label column.')
        if not df["true_label"].notna().all():
            raise ValueError('--shap_class=true requires complete true labels in the selected dataset.')
        return label_to_index(df["true_label"]).astype(np.int64)
    if class_arg not in CLASS_ORDER:
        raise ValueError(f"--shap_class must be pred, true, or a class name: {list(CLASS_ORDER)}")
    idx = int(list(CLASS_ORDER).index(class_arg))
    return np.full(len(df), idx, dtype=np.int64)


def _take_by_target(prob: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    return prob[np.arange(len(target_idx)), target_idx]


def _sample_for_plot(
    values: np.ndarray,
    shap_values: np.ndarray,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(values) <= max_samples:
        return values, shap_values
    rng = np.random.default_rng(20260515)
    idx = np.sort(rng.choice(len(values), size=max_samples, replace=False))
    return values[idx], shap_values[idx]


def _save_shap_like_beeswarm(
    *,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    out_prefix: Path,
    title: str,
    top_n: int,
    max_samples: int,
) -> None:
    if shap_values.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        _log(f"[SHAP] Skipping beeswarm plot because Matplotlib is unavailable: {e}")
        return

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    if top_n > 0:
        order = order[:top_n]
    names = [feature_names[i] for i in order]
    sv = shap_values[:, order]
    fv = feature_values[:, order]
    fv, sv = _sample_for_plot(fv, sv, max_samples=max_samples)

    n_feat = len(names)
    fig_h = max(4.0, 0.38 * n_feat + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    rng = np.random.default_rng(20260515)
    last_scatter = None
    for row_idx in range(n_feat):
        col = n_feat - 1 - row_idx
        x = sv[:, col]
        y = np.full_like(x, row_idx, dtype=np.float64) + rng.normal(0, 0.07, size=len(x))
        color = fv[:, col]
        last_scatter = ax.scatter(
            x,
            y,
            c=color,
            cmap="coolwarm",
            s=13,
            alpha=0.78,
            edgecolors="none",
            vmin=0.0,
            vmax=1.0,
        )

    ax.axvline(0, color="#888888", linewidth=1.0)
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("SHAP value (impact on ensemble output)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.28)
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=ax, pad=0.02)
        cbar.set_label("Feature value")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
    fig.tight_layout()
    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _write_contribution_tables(
    *,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    nominal_weights: Sequence[float],
    out_csv: Path,
) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total = float(np.sum(mean_abs))
    rows = []
    for i, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "nominal_weight": float(nominal_weights[i]),
                "mean_abs_shap": float(mean_abs[i]),
                "contribution_share": float(mean_abs[i] / total) if total > 0 else 0.0,
                "mean_feature_value": float(np.mean(feature_values[:, i])),
                "std_feature_value": float(np.std(feature_values[:, i])),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    cols = ["rank", "feature", "nominal_weight", "mean_abs_shap", "contribution_share", "mean_feature_value", "std_feature_value"]
    out = out[cols]
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def _compare_with_saved_predictions(
    *,
    out_dir: Path,
    tag: str,
    method: str,
    dataset: str,
    y_score: np.ndarray,
) -> Optional[Dict]:
    pred_root = out_dir / "03_ablation_ensemble" / "combos" / tag / method
    ds = safe_name((dataset or "").upper())
    if ds == "INTERNAL_OOF":
        pred_csv = pred_root / "internal_oof_predictions.csv"
    elif ds == "EXTERNAL_MERGE":
        pred_csv = pred_root / "external_predictions__EXTERNAL_MERGE.csv"
    else:
        pred_csv = pred_root / f"external_predictions__{ds}.csv"
    if not pred_csv.exists():
        return None
    old = pd.read_csv(pred_csv)
    prob_cols = [f"{cls}_prob" for cls in CLASS_ORDER]
    if not all(c in old.columns for c in prob_cols) or len(old) != len(y_score):
        return {"saved_predictions": pred_csv.as_posix(), "comparable": False}
    old_score = old[prob_cols].to_numpy(dtype=np.float64)
    return {
        "saved_predictions": pred_csv.as_posix(),
        "comparable": True,
        "max_abs_diff": float(np.max(np.abs(old_score - y_score))),
        "mean_abs_diff": float(np.mean(np.abs(old_score - y_score))),
    }


def _run_shap_contribution(
    *,
    csv_paths: List[Path],
    out_dir: Path,
    combo_models: Sequence[str],
    tag: str,
    run_id: str,
    do_intersection: bool,
    methods: Sequence[str],
    dataset: str,
    class_arg: str,
    top_n: int,
    max_samples: int,
) -> None:
    _log(f"[SHAP] Loading per-MIL predictions for tag={tag} | models={'+'.join(combo_models)}")
    detail_df, bm_to_mils = _load_combo_mil_detail(
        csv_paths,
        combo_models,
        do_intersection=do_intersection,
    )
    p_mean, p_vote, bm_probs, bm_picks = _build_combo_scores_from_mil_detail(detail_df, combo_models, bm_to_mils)

    method_scores = {"mean": p_mean, "vote": p_vote}
    for method in methods:
        if method not in method_scores:
            raise ValueError(f"--shap_method must be mean, vote, or both. Received: {method}")
        y_score_full = method_scores[method]
        ds_df, y_score = _select_explain_dataset(detail_df, y_score_full, dataset)
        target_idx = _target_class_index(ds_df, y_score, class_arg)
        out_root = out_dir / "05_shap_contribution" / tag / method / safe_name(dataset.upper())
        ensure_dir(out_root)

        # Attribute the mean score exactly and decompose vote support at the backbone level.
        big_names = list(combo_models)
        big_values: List[np.ndarray] = []
        if method == "mean":
            for bm in big_names:
                prob = bm_probs[bm][ds_df["__source_index"].to_numpy()]
                big_values.append(_take_by_target(prob, target_idx))
            big_x = np.stack(big_values, axis=1)
            big_shap = (big_x - np.mean(big_x, axis=0, keepdims=True)) / float(len(big_names))
        else:
            for bm in big_names:
                picks = bm_picks[bm][ds_df["__source_index"].to_numpy()]
                big_values.append((picks == target_idx).astype(np.float64))
            big_x = np.stack(big_values, axis=1)
            big_shap = (big_x - np.mean(big_x, axis=0, keepdims=True)) / float(len(big_names))
        big_weights = [1.0 / float(len(big_names)) for _ in big_names]
        big_table = _write_contribution_tables(
            shap_values=big_shap,
            feature_values=big_x,
            feature_names=big_names,
            nominal_weights=big_weights,
            out_csv=out_root / "big_model_contribution.csv",
        )
        _save_shap_like_beeswarm(
            shap_values=big_shap,
            feature_values=big_x,
            feature_names=big_names,
            out_prefix=out_root / "big_model_shap_beeswarm",
            title=f"{tag} | {method} | big-model contribution | class={class_arg}",
            top_n=top_n,
            max_samples=max_samples,
        )

        # Decompose the nested mean or target-class vote support by MIL model.
        mil_names: List[str] = []
        mil_weights: List[float] = []
        mil_values: List[np.ndarray] = []
        for bm in combo_models:
            mils = bm_to_mils[bm]
            for mil in mils:
                mid = safe_name(f"{bm}__{mil}")
                mil_names.append(f"{bm}__{mil}")
                mil_weights.append(1.0 / float(len(combo_models) * len(mils)))
                if method == "mean":
                    cols = [f"{mid}__{cls}" for cls in CLASS_ORDER]
                    prob = ds_df[cols].to_numpy(dtype=np.float64)
                    mil_values.append(_take_by_target(prob, target_idx))
                else:
                    cols = [f"{mid}__{cls}" for cls in CLASS_ORDER]
                    prob = ds_df[cols].to_numpy(dtype=np.float64)
                    pick = np.argmax(prob, axis=1)
                    mil_values.append((pick == target_idx).astype(np.float64))
        mil_x = np.stack(mil_values, axis=1)
        mil_shap = (mil_x - np.mean(mil_x, axis=0, keepdims=True)) * np.asarray(mil_weights, dtype=np.float64).reshape(1, -1)
        mil_table = _write_contribution_tables(
            shap_values=mil_shap,
            feature_values=mil_x,
            feature_names=mil_names,
            nominal_weights=mil_weights,
            out_csv=out_root / "mil_contribution.csv",
        )
        _save_shap_like_beeswarm(
            shap_values=mil_shap,
            feature_values=mil_x,
            feature_names=mil_names,
            out_prefix=out_root / "mil_shap_beeswarm",
            title=f"{tag} | {method} | MIL contribution | class={class_arg}",
            top_n=top_n,
            max_samples=max_samples,
        )

        pred = np.argmax(y_score, axis=1)
        reproduced = ds_df[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in ds_df.columns else [])].copy()
        for i, cls in enumerate(CLASS_ORDER):
            reproduced[f"{cls}_prob"] = y_score[:, i]
        reproduced["pred_label"] = [CLASS_ORDER[i] for i in pred]
        reproduced.to_csv(out_root / "reproduced_predictions.csv", index=False, encoding="utf-8-sig")

        compare = _compare_with_saved_predictions(
            out_dir=out_dir,
            tag=tag,
            method=method,
            dataset=dataset,
            y_score=y_score,
        )
        meta = {
            "tag": tag,
            "run_id": run_id,
            "method": method,
            "dataset": dataset,
            "class": class_arg,
            "n_samples": int(len(ds_df)),
            "combo_models": list(combo_models),
            "bm_to_mils": bm_to_mils,
            "top_big_models": big_table.head(10).to_dict(orient="records"),
            "top_mils": mil_table.head(10).to_dict(orient="records"),
            "saved_prediction_compare": compare,
            "note": 'Mean contributions exactly decompose the two-level average. Vote contributions represent support for the target class.',
        }
        with open(out_root / "shap_contribution_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _log(
            f"[SHAP] {method}/{dataset} completed: "
            f"big_top={big_table.iloc[0]['feature']} ({big_table.iloc[0]['contribution_share']:.3f}) | "
            f"mil_top={mil_table.iloc[0]['feature']} ({mil_table.iloc[0]['contribution_share']:.3f}) | "
            f"out={out_root.as_posix()}"
        )


def _write_metrics_no_plots(*, y_true: np.ndarray, y_score: np.ndarray, out_dir: Path) -> Dict:
    ensure_dir(out_dir)
    y_pred = np.argmax(y_score, axis=1)
    overall = metrics_mod.compute_overall_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))
    classwise = metrics_mod.compute_classwise_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))

    def _to_jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _to_jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        return v

    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(overall), f, ensure_ascii=False, indent=2)
    with open(out_dir / "classwise.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(classwise), f, ensure_ascii=False, indent=2)
    return overall


def _save_predictions_and_metrics(
    *,
    merged: pd.DataFrame,
    y_score: np.ndarray,
    out_root: Path,
    metrics_root: Path,
    title_suffix: str,
    no_plots: bool = False,
) -> Dict[str, Optional[Dict]]:
    ensure_dir(out_root)
    ensure_dir(metrics_root)

    # Require a zero-based index aligned with the prediction rows.
    base_cols = ["file_id", "fold", "dataset_type"]
    has_label = "true_label" in merged.columns
    keep = base_cols + (["true_label"] if has_label else [])
    df_all = merged[keep].copy()

    # INTERNAL OOF
    internal = df_all[df_all["dataset_type"].apply(is_internal)].copy()
    if len(internal) == 0:
        raise ValueError('No rows with dataset_type=INTERNAL were found.')
    if not has_label:
        raise ValueError('INTERNAL true labels are required for metrics and plots.')

    idx_internal = internal.index.to_numpy()
    p_int = y_score[idx_internal]
    y_int = label_to_index(internal["true_label"])
    pred_int = np.argmax(p_int, axis=1)

    internal_pred = internal[["file_id", "fold", "dataset_type", "true_label"]].copy()
    for i, cls in enumerate(CLASS_ORDER):
        internal_pred[f"{cls}_prob"] = p_int[:, i]
    internal_pred["pred_label"] = [CLASS_ORDER[i] for i in pred_int]
    internal_pred["is_correct"] = internal_pred["pred_label"].astype(str) == internal_pred["true_label"].astype(str)
    internal_pred.to_csv(out_root / "internal_oof_predictions.csv", index=False)

    # Return overall metrics for the internal and merged external summary.
    internal_overall: Optional[Dict] = None
    external_merge_overall: Optional[Dict] = None

    if no_plots:
        internal_overall = _write_metrics_no_plots(y_true=y_int, y_score=p_int, out_dir=metrics_root / "INTERNAL_OOF")
    else:
        save_metrics_bundle(
            y_true=y_int,
            y_score=p_int,
            out_dir=metrics_root / "INTERNAL_OOF",
            title_suffix=f"{title_suffix}_INTERNAL_OOF",
        )
        internal_overall = metrics_mod.compute_overall_metrics(y_int, np.argmax(p_int, axis=1), p_int, list(CLASS_ORDER))

    # Write external predictions per dataset and compute metrics when labels are present.
    externals = df_all[df_all["dataset_type"].apply(is_external)].copy()
    if len(externals) > 0:
        externals = externals.drop_duplicates(subset=["file_id", "dataset_type"]).copy()

        merged_ext_preds: List[pd.DataFrame] = []
        merged_ext_scores: List[np.ndarray] = []

        for dt, g in externals.groupby("dataset_type"):
            dt_name = safe_name(str(dt))
            idx = g.index.to_numpy()
            p = y_score[idx]
            pred = np.argmax(p, axis=1)

            cols = ["file_id", "fold", "dataset_type"] + (["true_label"] if has_label else [])
            out_df = g[cols].copy()
            for i, cls in enumerate(CLASS_ORDER):
                out_df[f"{cls}_prob"] = p[:, i]
            out_df["pred_label"] = [CLASS_ORDER[i] for i in pred]
            if has_label:
                out_df["is_correct"] = out_df["pred_label"].astype(str) == out_df["true_label"].astype(str)

            out_df.to_csv(out_root / f"external_predictions__{dt_name}.csv", index=False)

            if has_label and out_df["true_label"].notna().all():
                y = label_to_index(out_df["true_label"])
                if no_plots:
                    _write_metrics_no_plots(y_true=y, y_score=p, out_dir=metrics_root / dt_name)
                else:
                    save_metrics_bundle(
                        y_true=y,
                        y_score=p,
                        out_dir=metrics_root / dt_name,
                        title_suffix=f"{title_suffix}_{dt_name}",
                    )
                merged_ext_preds.append(out_df)
                merged_ext_scores.append(p)

        # EXTERNAL_MERGE
        if merged_ext_preds and merged_ext_scores:
            ext_merge_pred = pd.concat(merged_ext_preds, axis=0, ignore_index=True)
            ext_merge_pred.to_csv(out_root / "external_predictions__EXTERNAL_MERGE.csv", index=False)
            p_merge = np.concatenate(merged_ext_scores, axis=0)
            y_merge = label_to_index(ext_merge_pred["true_label"])
            if no_plots:
                external_merge_overall = _write_metrics_no_plots(
                    y_true=y_merge, y_score=p_merge, out_dir=metrics_root / "EXTERNAL_MERGE"
                )
            else:
                save_metrics_bundle(
                    y_true=y_merge,
                    y_score=p_merge,
                    out_dir=metrics_root / "EXTERNAL_MERGE",
                    title_suffix=f"{title_suffix}_EXTERNAL_MERGE",
                )
                external_merge_overall = metrics_mod.compute_overall_metrics(
                    y_merge, np.argmax(p_merge, axis=1), p_merge, list(CLASS_ORDER)
                )

    return {"internal_overall": internal_overall, "external_merge_overall": external_merge_overall}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default=r"CSV_data\12 models", help='Directory containing model prediction CSVs. Searched recursively.')
    p.add_argument("--output_dir", type=str, default=r"Ensemble_Results\_grid_test\all models combo", help='Output directory for predictions and metrics')
    p.add_argument("--min_models", type=int, default=2, help='Minimum number of models. Defaults to 2.')
    p.add_argument("--intersection", action="store_true", help='Keep only samples with predictions from every model')
    p.add_argument("--no_intersection", action="store_true", help='Allow incomplete predictions')
    p.add_argument("--combo_search", default=True, action="store_true", help='Enumerate backbone combinations after aggregating MIL models within each backbone')
    p.add_argument("--k_min", type=int, default=0, help='Minimum combination size. Defaults to --min_models.')
    p.add_argument("--k_max", type=int, default=0, help='Maximum combination size. Defaults to the number of backbones.')
    p.add_argument("--max_combos", type=int, default=0, help='Maximum number of combinations. Set 0 for no limit.')
    p.add_argument("--rank_by", type=str, default="internal_auc_macro_mean", help='Metric column used to rank summary_human.csv')
    p.add_argument("--no_plots", default=True, help='Write JSON and CSV metrics without PDF plots')
    p.add_argument("--shap", default=False, help='Explain the selected combination using SHAP-style contributions')
    p.add_argument("--shap_combo", type=str, default="98ff61d0", help='Run ID, tag, or list of models to explain. Defaults to the highest-ranked run.')
    p.add_argument("--shap_method", type=str, default="both", help='Contribution method: mean, vote, or both')
    p.add_argument("--shap_dataset", type=str, default="INTERNAL_OOF", help='Dataset to explain: INTERNAL_OOF, EXTERNAL_MERGE, or a dataset_type')
    p.add_argument("--shap_class", type=str, default="pred", help='Class to explain: pred, true, or a class name')
    p.add_argument("--shap_top_n", type=int, default=30, help='Contributions to show in the beeswarm plot. Set 0 to show all.')
    p.add_argument("--shap_max_samples", type=int, default=2000, help='Maximum beeswarm samples. Set 0 to use all samples.')
    p.add_argument("--shap_also_search", action="store_true", help='Continue combination search after running the explanation')
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    csv_paths = _discover_csvs(input_dir)
    _log(f"Found {len(csv_paths)} CSVs in input_dir={input_dir.as_posix()}）")
    if len(csv_paths) < int(args.min_models):
        raise ValueError(f"Too few models: found={len(csv_paths)} < --min_models={args.min_models}")

    # Keep samples with complete predictions from every model.
    do_intersection = True
    if args.no_intersection:
        do_intersection = False
    if args.intersection:
        do_intersection = True

    # Aggregate the MIL models within each backbone before combination search.
    merged_base, big_models = _aggregate_big_models_from_mils(
        csv_paths,
        do_intersection=do_intersection,
    )
    _log(f"Backbone aggregation completed: backbones={len(big_models)} | rows={len(merged_base)}")

    if args.shap:
        shap_methods = [m.strip().lower() for m in _parse_list_arg(args.shap_method)]
        if "both" in shap_methods:
            shap_methods = ["mean", "vote"]
        shap_methods = [m for m in shap_methods if m in {"mean", "vote"}]
        if not shap_methods:
            raise ValueError('--shap_method must be mean, vote, or both.')

        combo_models, tag, run_id, _ = _resolve_combo_for_explain(
            combo_arg=args.shap_combo,
            summary_path=out_dir / "summary_human.csv",
            big_models=big_models,
        )
        _run_shap_contribution(
            csv_paths=csv_paths,
            out_dir=out_dir,
            combo_models=combo_models,
            tag=tag,
            run_id=run_id,
            do_intersection=do_intersection,
            methods=shap_methods,
            dataset=args.shap_dataset,
            class_arg=args.shap_class,
            top_n=int(args.shap_top_n),
            max_samples=int(args.shap_max_samples),
        )
        if not args.shap_also_search:
            _log(f"done. shap output_dir={(out_dir / '05_shap_contribution' / tag).resolve()}")
            return

    base_cols = ["file_id", "fold", "dataset_type"]
    has_label = "true_label" in merged_base.columns
    if has_label:
        base_cols = base_cols + ["true_label"]

    def _run_combo(combo_models: List[str], tag_dir: Path, title_tag: str) -> Dict[str, Dict[str, Optional[Dict]]]:
        # Select only the columns needed for this combination.
        need_cols: List[str] = []
        for bm in combo_models:
            need_cols.extend([f"{bm}__{cls}" for cls in CLASS_ORDER])
            need_cols.append(f"{bm}__pick")
        df = merged_base[base_cols + need_cols].copy()

        # Require complete predictions from the models in this combination.
        if do_intersection:
            before = len(df)
            df = df.dropna(subset=need_cols).copy()
            after = len(df)
            if after == 0:
                raise ValueError('No samples have complete predictions for this combination.')
            if after != before:
                _log(f"[{title_tag}] intersection: {before} -> {after}")

        df = df.reset_index(drop=True)

        # Average backbone-level probabilities.
        mats = [df[[f"{bm}__{cls}" for cls in CLASS_ORDER]].to_numpy(dtype=np.float64) for bm in combo_models]
        p_models = np.stack(mats, axis=1)  # (n, k, c)
        p_mean = _renorm(np.mean(p_models, axis=1))

        out_mean = _save_predictions_and_metrics(
            merged=df[base_cols].copy(),
            y_score=p_mean,
            out_root=out_dir / "03_ablation_ensemble" / tag_dir / "mean",
            metrics_root=out_dir / "04_metrics" / "ablation_ensemble" / tag_dir / "mean",
            title_suffix=f"ablation_mean_{title_tag}",
            no_plots=bool(args.no_plots),
        )

        # Vote using the winning class from each backbone.
        picks = np.stack([df[f"{bm}__pick"].to_numpy(dtype=np.int64) for bm in combo_models], axis=1)  # (n, k)
        n, k = picks.shape
        c = len(CLASS_ORDER)
        counts = np.zeros((n, c), dtype=np.int64)
        for cls_idx in range(c):
            counts[:, cls_idx] = np.sum(picks == cls_idx, axis=1)
        vote_frac = _renorm(counts.astype(np.float64) / float(max(k, 1)))

        out_vote = _save_predictions_and_metrics(
            merged=df[base_cols].copy(),
            y_score=vote_frac,
            out_root=out_dir / "03_ablation_ensemble" / tag_dir / "vote",
            metrics_root=out_dir / "04_metrics" / "ablation_ensemble" / tag_dir / "vote",
            title_suffix=f"ablation_vote_{title_tag}",
            no_plots=bool(args.no_plots),
        )

        return {"mean": out_mean, "vote": out_vote}

    if not args.combo_search:
        _log(f"Combination search disabled. Using every backbone (n={len(big_models)}）")
        _run_combo(big_models, tag_dir=Path(), title_tag="all_big_models")
    else:
        # Enumerate combinations.
        k_min = int(args.k_min) if int(args.k_min) > 0 else int(args.min_models)
        k_max = int(args.k_max) if int(args.k_max) > 0 else len(big_models)
        k_min = max(1, min(k_min, len(big_models)))
        k_max = max(k_min, min(k_max, len(big_models)))

        combos: List[Tuple[str, ...]] = []
        for k in range(k_min, k_max + 1):
            combos.extend(list(itertools.combinations(big_models, k)))
        if int(args.max_combos) > 0:
            combos = combos[: int(args.max_combos)]

        n_total = len(combos)
        _log(f"Combination search enabled. Combinations={n_total}（k_min={k_min}, k_max={k_max}；no_plots={bool(args.no_plots)}）")

        rows: List[Dict] = []
        for idx, combo in enumerate(combos, start=1):
            combo_list = list(combo)
            combo_key = "+".join(combo_list)
            combo_hash = _short_hash(combo_key)
            tag = f"combo_k{len(combo_list)}_{combo_hash}"
            tag_dir = Path("combos") / tag

            bar = _progress_bar(idx, n_total)
            _log(f"{bar} {idx}/{n_total} | {tag} | models={combo_key}")

            start_ts = time.time()
            start_str = time.strftime("%Y-%m-%d %H:%M:%S")
            status = "success"
            err_msg = ""
            out_metrics: Dict[str, Dict[str, Optional[Dict]]] = {"mean": {}, "vote": {}}
            try:
                out_metrics = _run_combo(combo_list, tag_dir=tag_dir, title_tag=tag)
            except Exception as e:
                status = "failed"
                err_msg = str(e)
                _log(f"[{tag}] failed: {err_msg}")
            end_str = time.strftime("%Y-%m-%d %H:%M:%S")
            dur = time.time() - start_ts

            def _pick(m: Optional[Dict], key: str) -> Optional[float]:
                if not m:
                    return None
                v = m.get(key, None)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            mean_int = (out_metrics.get("mean") or {}).get("internal_overall")
            mean_ext = (out_metrics.get("mean") or {}).get("external_merge_overall")
            vote_int = (out_metrics.get("vote") or {}).get("internal_overall")
            vote_ext = (out_metrics.get("vote") or {}).get("external_merge_overall")

            rows.append(
                {
                    "rank": None,
                    "run_id": combo_hash,
                    "状态": status,
                    "开始时间": start_str,
                    "结束时间": end_str,
                    "输出目录": str(out_dir / "03_ablation_ensemble" / tag_dir),
                    "输入目录": str(input_dir),
                    "耗时(s)": round(dur, 3),
                    "k": len(combo_list),
                    "models": combo_key,
                    "tag": tag,
                    "internal_auc_macro_mean": _pick(mean_int, "auc_macro"),
                    "internal_auc_micro_mean": _pick(mean_int, "auc_micro"),
                    "internal_accuracy_mean": _pick(mean_int, "accuracy"),
                    "internal_balanced_accuracy_mean": _pick(mean_int, "balanced_accuracy"),
                    "external_auc_macro_mean": _pick(mean_ext, "auc_macro"),
                    "external_auc_micro_mean": _pick(mean_ext, "auc_micro"),
                    "external_accuracy_mean": _pick(mean_ext, "accuracy"),
                    "external_balanced_accuracy_mean": _pick(mean_ext, "balanced_accuracy"),
                    "has_external_merge_mean": int(mean_ext is not None),
                    "internal_auc_macro_vote": _pick(vote_int, "auc_macro"),
                    "internal_auc_micro_vote": _pick(vote_int, "auc_micro"),
                    "internal_accuracy_vote": _pick(vote_int, "accuracy"),
                    "internal_balanced_accuracy_vote": _pick(vote_int, "balanced_accuracy"),
                    "external_auc_macro_vote": _pick(vote_ext, "auc_macro"),
                    "external_auc_micro_vote": _pick(vote_ext, "auc_micro"),
                    "external_accuracy_vote": _pick(vote_ext, "accuracy"),
                    "external_balanced_accuracy_vote": _pick(vote_ext, "balanced_accuracy"),
                    "has_external_merge_vote": int(vote_ext is not None),
                    "error": err_msg,
                }
            )

        summary_df = pd.DataFrame(rows)
        rank_col = str(args.rank_by)
        if rank_col not in summary_df.columns:
            _log(f"Ranking column not found: {rank_col}. Using internal_auc_macro_mean.")
            rank_col = "internal_auc_macro_mean"

        ok_mask = summary_df["状态"].astype(str) == "success"
        summary_df.loc[~ok_mask, rank_col] = np.nan
        summary_df = summary_df.sort_values(by=rank_col, ascending=False, na_position="last").reset_index(drop=True)
        summary_df["rank"] = np.arange(1, len(summary_df) + 1, dtype=int)

        summary_path = out_dir / "summary_human.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        _log(f"Wrote summary_human.csv: {summary_path.as_posix()}（rank_by={rank_col}）")

    _log(f"done. output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()


