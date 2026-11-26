"""
Run all three Hybrid-B stages and evaluate selected or enumerated backbone combinations.
Step 0 caches softmax probabilities. Step 1 prepares common prediction tables. Step 2 runs nested cross-validation and external inference. The result directories and run IDs match the mean and voting ensembles from stepX_avg_vote.py, so stepY_render_combo_reports.py can render all methods for the same run ID.
Use --mode search for combinations, or --mode single with --combo CHIEF+UNI+CONCH to evaluate one combination. Single mode without --combo uses all available backbones.
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

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import metrics as metrics_mod
from hybridb_common import (
    CLASS_ABBR,
    ModelRef,
    build_stage1_table_for_backbone,
    discover_model_csvs,
    ensure_dir,
    fit_predict_stage1_ovr,
    get_prob_feature_cols,
    is_external,
    is_internal,
    label_to_index,
    merge_all_models,
    parse_mils_arg,
    safe_name,
    save_metrics_bundle,
    stage1_inner_oof_for_backbone,
    stage2_multinomial_lr,
)
from softmax_probs import CLASS_ORDER, run_softmax


# Share directory names with stepX_avg_vote.py.
# Write hybridb beside mean and vote under 03_ablation_ensemble/<tag>/<method>.
# The report renderer supports all three methods.
PRED_ROOT_NAME = "03_ablation_ensemble"
METRICS_GROUP_NAME = "ablation_ensemble"
METHOD_NAME = "hybridb"


# Input parsing and common helpers

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}][HybridB] {msg}", flush=True)


def _short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def _progress_bar(i: int, n: int, width: int = 28) -> str:
    if n <= 0:
        return "[" + ("?" * width) + "]"
    i = max(0, min(i, n))
    filled = int(round(width * (i / n)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _parse_list_arg(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    for sep in [";", ",", "+"]:
        x = x.replace(sep, ",")
    return [p.strip() for p in x.split(",") if p.strip()]


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


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _write_metrics_no_plots(*, y_true: np.ndarray, y_score: np.ndarray, out_dir: Path) -> Dict:
    """Write overall and classwise JSON metrics without plotting."""
    ensure_dir(out_dir)
    y_pred = np.argmax(y_score, axis=1)
    overall = metrics_mod.compute_overall_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))
    classwise = metrics_mod.compute_classwise_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))
    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(overall), f, ensure_ascii=False, indent=2)
    with open(out_dir / "classwise.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(classwise), f, ensure_ascii=False, indent=2)
    return overall


# Step 0 probability conversion and caching

def _prepare_probabilities(
    refs: Sequence[ModelRef],
    *,
    probability_dir: Path,
    force: bool = False,
    quiet: bool = True,
) -> Dict[str, Path]:
    """Cache softmax probabilities for the base-model CSVs."""
    ensure_dir(probability_dir)
    model_to_path: Dict[str, Path] = {}
    total = len(refs)
    t0_all = time.time()
    skipped = 0
    processed = 0
    _log(f"Step 0 probability conversion started (force={force}, total={total}）")
    for ref in refs:
        out_csv = probability_dir / f"{ref.model_id}.csv"
        model_to_path[ref.model_id] = out_csv
        if out_csv.exists() and (not force):
            skipped += 1
            if not quiet:
                _log(f"  skip ({processed + skipped}/{total}) {ref.model_id}")
            continue
        processed += 1
        if not quiet:
            _log(f"  run  ({processed + skipped}/{total}) {ref.model_id}")
        run_softmax(
            input_csv=str(ref.path),
            output_csv=str(out_csv),
        )
    _log(f"Step 0 completed (processed={processed}, skipped={skipped}, total={total}, cost={time.time() - t0_all:.1f}s）")
    return model_to_path


# Precompute reusable Stage 1 predictions

class _HybridBCache:
    """Cache Stage 1 OOF and validation predictions for each outer fold, full-internal models and OOF predictions, and per-dataset external predictions. This lets each combination fit and evaluate only Stage 2."""

    def __init__(self) -> None:
        self.outer_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        self.stage1_oof_train: Dict[str, List[np.ndarray]] = {}
        self.stage1_pred_val: Dict[str, List[np.ndarray]] = {}
        self.stage1_full_oof: Dict[str, np.ndarray] = {}
        self.stage1_full_models: Dict[str, Dict[str, LogisticRegression]] = {}
        self.stage1_full_ext: Dict[str, Dict[str, np.ndarray]] = {}
        self.ext_meta: Dict[str, pd.DataFrame] = {}


def _precompute_hybridb_cache(
    *,
    internal: pd.DataFrame,
    externals_per_dt: Dict[str, pd.DataFrame],
    backbones: Sequence[str],
    mils: Sequence[str],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    stage1_C: float,
    stage1_class_weight: Optional[str],
) -> Tuple[_HybridBCache, np.ndarray]:
    cache = _HybridBCache()
    y_internal = label_to_index(internal["true_label"])

    skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    cache.outer_splits = list(skf.split(np.zeros(len(internal)), y_internal))

    n_internal = len(internal)
    n_class = len(CLASS_ORDER)
    abbr_cols_pattern = [f"_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER]

    _log(
        f"Precomputing Stage 1 for backbones={len(backbones)} | mils={len(mils)} | "
        f"outer={outer_folds} | inner={inner_folds} | n_internal={n_internal}"
    )

    for bi, bb in enumerate(backbones, start=1):
        t0 = time.time()
        cache.stage1_oof_train[bb] = []
        cache.stage1_pred_val[bb] = []

        # Generate inner-CV OOF predictions in the training subset and predict the outer validation subset.
        for k_idx, (tr_idx, va_idx) in enumerate(cache.outer_splits):
            df_tr = internal.iloc[tr_idx].copy()
            df_va = internal.iloc[va_idx].copy()
            tbl_tr = build_stage1_table_for_backbone(df_tr, backbone=bb, mils=list(mils))
            tbl_va = build_stage1_table_for_backbone(df_va, backbone=bb, mils=list(mils))

            oof_bb = stage1_inner_oof_for_backbone(
                df_train=tbl_tr,
                backbone=bb,
                mils=list(mils),
                inner_folds=inner_folds,
                seed=seed,
                C=stage1_C,
                class_weight=stage1_class_weight,
            )
            pred_va_bb, _ = fit_predict_stage1_ovr(
                df_train=tbl_tr,
                df_pred=tbl_va,
                backbone=bb,
                mils=list(mils),
                C=stage1_C,
                class_weight=stage1_class_weight,
            )
            # Collect four-class fold probabilities.
            cache.stage1_oof_train[bb].append(
                oof_bb[[f"{bb}{p}" for p in abbr_cols_pattern]].to_numpy(dtype=np.float32)
            )
            cache.stage1_pred_val[bb].append(
                pred_va_bb[[f"{bb}{p}" for p in abbr_cols_pattern]].to_numpy(dtype=np.float32)
            )

        # Generate full-internal OOF predictions and fit on all internal cases.
        tbl_int = build_stage1_table_for_backbone(internal, backbone=bb, mils=list(mils))
        oof_full = stage1_inner_oof_for_backbone(
            df_train=tbl_int,
            backbone=bb,
            mils=list(mils),
            inner_folds=inner_folds,
            seed=seed,
            C=stage1_C,
            class_weight=stage1_class_weight,
        )
        cache.stage1_full_oof[bb] = oof_full[[f"{bb}{p}" for p in abbr_cols_pattern]].to_numpy(dtype=np.float32)
        _, models_full = fit_predict_stage1_ovr(
            df_train=tbl_int,
            df_pred=tbl_int.iloc[:1].copy(),
            backbone=bb,
            mils=list(mils),
            C=stage1_C,
            class_weight=stage1_class_weight,
        )
        cache.stage1_full_models[bb] = models_full

        # Predict each external dataset.
        cache.stage1_full_ext[bb] = {}
        for dt_name, g in externals_per_dt.items():
            if len(g) == 0:
                continue
            tbl_ext = build_stage1_table_for_backbone(g, backbone=bb, mils=list(mils))
            out_mat = np.zeros((len(tbl_ext), n_class), dtype=np.float32)
            for i, cls in enumerate(CLASS_ORDER):
                ab = CLASS_ABBR[cls]
                feat = [f"{mil}_p_{ab}" for mil in mils]
                X = tbl_ext[feat].to_numpy(dtype=np.float32)
                if cls in models_full:
                    out_mat[:, i] = models_full[cls].predict_proba(X)[:, 1]
                else:
                    prior = float((internal["true_label"].astype(str) == cls).mean())
                    out_mat[:, i] = prior
            cache.stage1_full_ext[bb][dt_name] = out_mat

        _log(f"  [{bi}/{len(backbones)}] backbone={bb} precomputation complete ({time.time() - t0:.1f}s）")

    # Keep the external dataset join keys for output.
    for dt_name, g in externals_per_dt.items():
        meta_cols = ["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in g.columns else [])
        cache.ext_meta[dt_name] = g[meta_cols].reset_index(drop=True).copy()

    return cache, y_internal


# Hybrid-B prediction for one combination

def _stack_features(
    parts: Sequence[np.ndarray],
) -> np.ndarray:
    """Concatenate K backbone matrices of shape (n, 4) into a matrix of shape (n, 4 * K)."""
    return np.concatenate(list(parts), axis=1).astype(np.float32, copy=False)


def _run_hybridb_for_combo(
    *,
    combo_bbs: Sequence[str],
    internal: pd.DataFrame,
    y_internal: np.ndarray,
    cache: _HybridBCache,
    externals_per_dt: Dict[str, pd.DataFrame],
    stage2_C: float,
    save_fold_artifacts_dir: Optional[Path] = None,
    save_final_joblib_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Fit and evaluate Stage 2 for a backbone combination using cached Stage 1 predictions. Return internal OOF probabilities, external probabilities by dataset, and fitted artifacts."""
    combo_bbs = list(combo_bbs)
    n_internal = len(internal)
    p_oof = np.zeros((n_internal, len(CLASS_ORDER)), dtype=np.float64)

    fold_stage2_coefs: List[Dict[str, object]] = []

    for k_idx, (tr_idx, va_idx) in enumerate(cache.outer_splits):
        X_tr = _stack_features([cache.stage1_oof_train[bb][k_idx] for bb in combo_bbs])
        X_va = _stack_features([cache.stage1_pred_val[bb][k_idx] for bb in combo_bbs])
        y_tr = y_internal[tr_idx]

        stage2 = stage2_multinomial_lr(C=stage2_C)
        stage2.fit(X_tr, y_tr)
        p_va = stage2.predict_proba(X_va)
        p_oof[va_idx] = p_va

        if save_fold_artifacts_dir is not None:
            lr2: LogisticRegression = stage2.named_steps["lr"]
            feat_cols = []
            for bb in combo_bbs:
                feat_cols.extend([f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER])
            fold_stage2_coefs.append(
                {
                    "fold": k_idx + 1,
                    "feature_cols": feat_cols,
                    "intercept": lr2.intercept_.tolist(),
                    "coef": lr2.coef_.tolist(),
                    "classes": list(map(int, lr2.classes_)),
                    "C": stage2_C,
                }
            )

    # Fit the final model on all internal cases for external inference.
    X_full = _stack_features([cache.stage1_full_oof[bb] for bb in combo_bbs])
    stage2_final = stage2_multinomial_lr(C=stage2_C)
    stage2_final.fit(X_full, y_internal)

    ext_probs: Dict[str, np.ndarray] = {}
    for dt_name in externals_per_dt.keys():
        parts = []
        ok = True
        for bb in combo_bbs:
            mat = cache.stage1_full_ext.get(bb, {}).get(dt_name)
            if mat is None:
                ok = False
                break
            parts.append(mat)
        if not ok:
            continue
        X_ext = _stack_features(parts)
        ext_probs[dt_name] = stage2_final.predict_proba(X_ext)

    artifacts: Dict[str, object] = {}
    if save_fold_artifacts_dir is not None:
        ensure_dir(save_fold_artifacts_dir)
        with open(save_fold_artifacts_dir / "stage2_coefs_per_fold.json", "w", encoding="utf-8") as f:
            json.dump(fold_stage2_coefs, f, ensure_ascii=False, indent=2)
        # Final model coefficients
        lr_final: LogisticRegression = stage2_final.named_steps["lr"]
        feat_cols = []
        for bb in combo_bbs:
            feat_cols.extend([f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER])
        with open(save_fold_artifacts_dir / "stage2_coef_final.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_cols": feat_cols,
                    "intercept": lr_final.intercept_.tolist(),
                    "coef": lr_final.coef_.tolist(),
                    "classes": list(map(int, lr_final.classes_)),
                    "C": stage2_C,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        artifacts["fold_stage2_coefs"] = fold_stage2_coefs

    if save_final_joblib_path is not None:
        ensure_dir(save_final_joblib_path.parent)
        joblib.dump(
            {
                "combo_backbones": combo_bbs,
                "stage1_full_models": {bb: cache.stage1_full_models[bb] for bb in combo_bbs},
                "stage2_final": stage2_final,
            },
            save_final_joblib_path,
        )

    return p_oof, ext_probs, artifacts


# Prediction and metric outputs

def _save_predictions_and_metrics(
    *,
    internal: pd.DataFrame,
    y_internal: np.ndarray,
    p_oof: np.ndarray,
    ext_meta: Dict[str, pd.DataFrame],
    ext_probs: Dict[str, np.ndarray],
    out_root: Path,
    metrics_root: Path,
    title_suffix: str,
    no_plots: bool,
) -> Dict[str, Optional[Dict]]:
    """Write predictions and metrics using the layout shared by the mean and voting ensembles."""
    ensure_dir(out_root)
    ensure_dir(metrics_root)

    # ---- INTERNAL OOF ----
    internal_oof = internal[["file_id", "fold", "dataset_type", "true_label"]].copy()
    for i, cls in enumerate(CLASS_ORDER):
        internal_oof[f"{cls}_prob"] = p_oof[:, i]
    pred_int = np.argmax(p_oof, axis=1)
    internal_oof["pred_label"] = [CLASS_ORDER[i] for i in pred_int]
    internal_oof["is_correct"] = internal_oof["pred_label"].astype(str) == internal_oof["true_label"].astype(str)
    internal_oof.to_csv(out_root / "internal_oof_predictions.csv", index=False)

    if no_plots:
        internal_overall = _write_metrics_no_plots(
            y_true=y_internal, y_score=p_oof, out_dir=metrics_root / "INTERNAL_OOF"
        )
    else:
        save_metrics_bundle(
            y_true=y_internal,
            y_score=p_oof,
            out_dir=metrics_root / "INTERNAL_OOF",
            title_suffix=f"{title_suffix}_INTERNAL_OOF",
        )
        internal_overall = metrics_mod.compute_overall_metrics(
            y_internal, pred_int, p_oof, list(CLASS_ORDER)
        )

    # Per-dataset and pooled external predictions
    external_merge_overall: Optional[Dict] = None
    merged_ext_preds: List[pd.DataFrame] = []
    merged_ext_scores: List[np.ndarray] = []

    for dt_name in sorted(ext_meta.keys()):
        if dt_name not in ext_probs:
            continue
        g = ext_meta[dt_name]
        p = ext_probs[dt_name]
        if len(g) != len(p):
            _log(f"  [WARN] {dt_name}: metadata and prediction row counts differ ({len(g)} vs {len(p)}). Skipping.")
            continue
        pred = np.argmax(p, axis=1)
        has_label = "true_label" in g.columns
        out_df = g.copy()
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


# Resolve single-mode combinations

def _resolve_combo_single(
    *,
    combo_arg: str,
    backbones: Sequence[str],
    summary_path: Path,
) -> Tuple[List[str], str, str]:
    """Resolve --combo from a run ID, tag, or comma- or plus-separated backbone list. An empty value selects all backbones. Return (backbones, tag, run_id)."""
    bb_set = set(backbones)
    arg = (combo_arg or "").strip()
    backbones_sorted = sorted(backbones)
    if not arg:
        return list(backbones_sorted), "all_backbones", _short_hash("+".join(backbones_sorted))

    # Look up the combination in summary_human.csv first.
    if summary_path.exists():
        try:
            df = _read_csv_fallback(summary_path)
        except Exception as e:
            _log(f"Could not read summary_human.csv: {e}. Parsing --combo as a list.")
            df = None
        if df is not None and len(df) > 0:
            masks = []
            for col in ["run_id", "tag"]:
                if col in df.columns:
                    masks.append(df[col].astype(str) == arg)
            if masks:
                mask = masks[0]
                for m in masks[1:]:
                    mask = mask | m
                hit = df[mask].copy()
                if len(hit) > 0:
                    row = hit.iloc[0].to_dict()
                    models_str = str(row.get("models", "") or "").strip()
                    combo_models = _parse_list_arg(models_str)
                    if combo_models:
                        missing = [m for m in combo_models if m not in bb_set]
                        if missing:
                            raise ValueError(f"The matched summary row includes backbones missing from the inputs: {missing}")
                        # Sort before hashing so run IDs match search mode.
                        combo_models_sorted = sorted(combo_models)
                        tag = str(row.get("tag") or f"combo_k{len(combo_models_sorted)}_{_short_hash('+'.join(combo_models_sorted))}")
                        run_id = str(row.get("run_id") or _short_hash("+".join(combo_models_sorted)))
                        return combo_models_sorted, tag, run_id

    # Parse the input as a backbone list.
    combo_models = _parse_list_arg(arg)
    if not combo_models:
        raise ValueError('--combo did not name any backbones or match a summary row.')
    missing = [m for m in combo_models if m not in bb_set]
    if missing:
        raise ValueError(f"Unknown backbones in --combo: {missing}\nAvailable backbones: {list(backbones)}")
    # Sort before hashing so run IDs match search mode.
    combo_models = sorted(combo_models)
    run_id = _short_hash("+".join(combo_models))
    tag = f"combo_k{len(combo_models)}_{run_id}"
    return combo_models, tag, run_id


# Command-line entry point

def main() -> None:
    p = argparse.ArgumentParser()
    # IO
    p.add_argument("--input_dir", type=str, default=r"CSV_data\12 models",
                   help='Directory of model CSVs using the <MIL>/<Backbone>.csv layout')
    p.add_argument("--output_dir", type=str, default=r"outputs/ensemble/hybridb_combo",
                   help='Output root shared with stepX_avg_vote.py')
    p.add_argument("--mils", type=str, default="",
                   help='Comma-separated MIL names. Defaults to automatic discovery.')
    # Run mode
    p.add_argument("--mode", type=str, default="search", choices=["search", "single"],
                   help='Use search to enumerate combinations or single to evaluate one combination')
    p.add_argument("--combo", type=str, default="",
                   help='Backbone list, run ID, or tag for single mode. Defaults to all backbones.')
    # Hybrid-B hyperparameters
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--inner_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1_C", type=float, default=1.0)
    p.add_argument("--stage2_C", type=float, default=1.0)
    p.add_argument("--stage1_class_weight", type=str, default="balanced", help='Optional class weighting: balanced')
    # Step 0 probability conversion
    p.add_argument("--force_softmax", action="store_true", help='Recompute existing softmax outputs')
    p.add_argument("--quiet_softmax", action="store_true", default=True, help='Reduce output during probability conversion')
    # Combination search settings
    p.add_argument("--k_min", type=int, default=2, help='Minimum combination size. Defaults to 2.')
    p.add_argument("--k_max", type=int, default=0, help='Maximum combination size. Defaults to the number of backbones.')
    p.add_argument("--max_combos", type=int, default=0, help='Maximum number of combinations. Set 0 for no limit.')
    p.add_argument("--rank_by", type=str, default="internal_auc_macro_hybridb",
                   help='Metric column used to rank summary_human.csv')
    # Output controls
    p.add_argument("--no_plots", default=True, action="store_true",
                   help='Write JSON and CSV metrics without PDF figures')
    p.add_argument("--save_fold_artifacts", action="store_true",
                   help='Save Stage 2 fold coefficients for each combination')
    p.add_argument("--save_final_joblib", action="store_true",
                   help='Save the final model for each combination')
    p.add_argument("--save_stage1_inputs", action="store_true",
                   help='Save the joined and strict-intersection CSVs under 02_stage1/inputs')

    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    ensure_dir(out_dir)

    # Discover prediction CSVs and convert logits.
    mils_filter = parse_mils_arg(args.mils)
    refs = discover_model_csvs(input_dir, mils_filter=mils_filter)
    _log(f"Found {len(refs)} base-model CSVs in input_dir={input_dir.as_posix()}）")

    probability_dir = out_dir / "01_softmax" / "probability_csv"
    model_to_path = _prepare_probabilities(
        refs,
        probability_dir=probability_dir,
        force=bool(args.force_softmax),
        quiet=bool(args.quiet_softmax),
    )

    # Join probabilities and prepare the strict intersection.
    _log('Joining model probabilities and selecting the strict intersection...')
    merged = merge_all_models(model_to_path)
    prob_cols = get_prob_feature_cols(merged)
    clean = merged.dropna(subset=prob_cols).copy().reset_index(drop=True)

    inputs_dir = out_dir / "02_stage1" / "inputs"
    ensure_dir(inputs_dir)
    if args.save_stage1_inputs:
        merged.to_csv(inputs_dir / "merged_all.csv", index=False)
        clean.to_csv(inputs_dir / "clean_intersection.csv", index=False)

    internal = clean[clean["dataset_type"].apply(is_internal)].copy().reset_index(drop=True)
    if "true_label" not in internal.columns:
        raise ValueError('INTERNAL true labels are required for Stage 2 training and evaluation.')
    externals = clean[clean["dataset_type"].apply(is_external)].copy()
    if len(externals) > 0:
        externals = externals.drop_duplicates(subset=["file_id", "dataset_type"]).copy()

    # Normalize backbone and MIL names for filenames and run IDs.
    backbones_all = sorted({safe_name(r.backbone) for r in refs})
    mils_all = sorted({safe_name(r.mil) for r in refs})

    # Build the Stage 1 table using the original MIL and backbone names.
    # The probability columns use the <MIL>__<Backbone>__<CLASS> pattern.
    mils_for_stage1 = sorted({r.mil for r in refs})
    backbones_for_stage1 = sorted({r.backbone for r in refs})

    _log(f"Backbones: {backbones_for_stage1}")
    _log(f"mils  = {mils_for_stage1}")
    _log(
        f"clean_intersection={len(clean)} | internal={len(internal)} | external={len(externals)} | "
        f"prob_cols={len(prob_cols)}"
    )

    # Save metadata for downstream scripts.
    meta = {
        "mils": mils_for_stage1,
        "backbones": backbones_for_stage1,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "stage1_C": args.stage1_C,
        "stage2_C": args.stage2_C,
        "stage1_class_weight": args.stage1_class_weight.strip() or None,
        "n_internal": int(len(internal)),
        "n_external": int(len(externals)),
    }
    with open(out_dir / "02_stage1" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Split external data by dataset_type while preserving row order.
    externals_per_dt: Dict[str, pd.DataFrame] = {}
    if len(externals) > 0:
        for dt, g in externals.groupby("dataset_type"):
            externals_per_dt[safe_name(str(dt))] = g.reset_index(drop=True).copy()

    # Precompute Stage 1 once for all combinations.
    cache, y_internal = _precompute_hybridb_cache(
        internal=internal,
        externals_per_dt=externals_per_dt,
        backbones=backbones_for_stage1,
        mils=mils_for_stage1,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        stage1_C=args.stage1_C,
        stage1_class_weight=(args.stage1_class_weight.strip() or None),
    )

    # Map backbone names to safe names so run IDs match stepX_avg_vote.py.
    bb_raw_to_safe = {bb: safe_name(bb) for bb in backbones_for_stage1}
    bb_safe_to_raw = {safe_name(bb): bb for bb in backbones_for_stage1}
    backbones_safe = sorted(bb_safe_to_raw.keys())

    # Evaluate a single combination.
    if args.mode == "single":
        combo_safe, tag, run_id = _resolve_combo_single(
            combo_arg=args.combo,
            backbones=backbones_safe,
            summary_path=out_dir / "summary_human.csv",
        )
        combo_raw = [bb_safe_to_raw[s] for s in combo_safe]
        _log(f"Single mode: tag={tag} | run_id={run_id} | backbones={combo_safe}")

        if tag == "all_backbones":
            tag_dir = Path("all_backbones")
        else:
            tag_dir = Path("combos") / tag

        # Use the same output layout as stepX_avg_vote.py.
        out_root = out_dir / PRED_ROOT_NAME / tag_dir / METHOD_NAME
        metrics_root = out_dir / "04_metrics" / METRICS_GROUP_NAME / tag_dir / METHOD_NAME

        fold_dir = (out_root / "fold_artifacts") if args.save_fold_artifacts else None
        final_path = (out_root / "final_model_for_external" / "hybridb_final.joblib") if args.save_final_joblib else None

        t0 = time.time()
        p_oof, ext_probs, _ = _run_hybridb_for_combo(
            combo_bbs=combo_raw,
            internal=internal,
            y_internal=y_internal,
            cache=cache,
            externals_per_dt=externals_per_dt,
            stage2_C=args.stage2_C,
            save_fold_artifacts_dir=fold_dir,
            save_final_joblib_path=final_path,
        )
        _save_predictions_and_metrics(
            internal=internal,
            y_internal=y_internal,
            p_oof=p_oof,
            ext_meta=cache.ext_meta,
            ext_probs=ext_probs,
            out_root=out_root,
            metrics_root=metrics_root,
            title_suffix=f"hybridb_{tag}",
            no_plots=bool(args.no_plots),
        )
        _log(f"Single run completed ({time.time() - t0:.1f}s）：out_root={out_root.as_posix()}")
        _log(f"done. output_dir={out_dir.resolve()}")
        return

    # Enumerate backbone combinations in search mode.
    k_min = max(1, int(args.k_min))
    k_max = int(args.k_max) if int(args.k_max) > 0 else len(backbones_safe)
    k_min = min(k_min, len(backbones_safe))
    k_max = max(k_min, min(k_max, len(backbones_safe)))

    combos: List[Tuple[str, ...]] = []
    for k in range(k_min, k_max + 1):
        combos.extend(list(itertools.combinations(backbones_safe, k)))
    if int(args.max_combos) > 0:
        combos = combos[: int(args.max_combos)]

    n_total = len(combos)
    _log(
        f"Combination search over k in [{k_min},{k_max}] | n_combos={n_total} | no_plots={bool(args.no_plots)} | "
        f"save_fold_artifacts={bool(args.save_fold_artifacts)} | save_final_joblib={bool(args.save_final_joblib)}"
    )

    rows: List[Dict] = []
    for idx, combo in enumerate(combos, start=1):
        combo_safe = list(combo)
        combo_raw = [bb_safe_to_raw[s] for s in combo_safe]
        combo_key = "+".join(combo_safe)
        run_id = _short_hash(combo_key)
        tag = f"combo_k{len(combo_safe)}_{run_id}"
        tag_dir = Path("combos") / tag

        # Use the same output layout as stepX_avg_vote.py.
        out_root = out_dir / PRED_ROOT_NAME / tag_dir / METHOD_NAME
        metrics_root = out_dir / "04_metrics" / METRICS_GROUP_NAME / tag_dir / METHOD_NAME
        fold_dir = (out_root / "fold_artifacts") if args.save_fold_artifacts else None
        final_path = (out_root / "final_model_for_external" / "hybridb_final.joblib") if args.save_final_joblib else None

        bar = _progress_bar(idx, n_total)
        _log(f"{bar} {idx}/{n_total} | {tag} | k={len(combo_safe)} | models={combo_key}")

        start_ts = time.time()
        start_str = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "success"
        err_msg = ""
        out_metrics: Dict[str, Optional[Dict]] = {"internal_overall": None, "external_merge_overall": None}
        try:
            p_oof, ext_probs, _ = _run_hybridb_for_combo(
                combo_bbs=combo_raw,
                internal=internal,
                y_internal=y_internal,
                cache=cache,
                externals_per_dt=externals_per_dt,
                stage2_C=args.stage2_C,
                save_fold_artifacts_dir=fold_dir,
                save_final_joblib_path=final_path,
            )
            out_metrics = _save_predictions_and_metrics(
                internal=internal,
                y_internal=y_internal,
                p_oof=p_oof,
                ext_meta=cache.ext_meta,
                ext_probs=ext_probs,
                out_root=out_root,
                metrics_root=metrics_root,
                title_suffix=f"hybridb_{tag}",
                no_plots=bool(args.no_plots),
            )
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

        m_int = out_metrics.get("internal_overall")
        m_ext = out_metrics.get("external_merge_overall")
        rows.append(
            {
                "rank": None,
                "run_id": run_id,
                "状态": status,
                "开始时间": start_str,
                "结束时间": end_str,
                "输出目录": str(out_dir / PRED_ROOT_NAME / tag_dir),
                "输入目录": str(input_dir),
                "耗时(s)": round(dur, 3),
                "k": len(combo_safe),
                "models": combo_key,
                "tag": tag,
                "internal_auc_macro_hybridb": _pick(m_int, "auc_macro"),
                "internal_auc_micro_hybridb": _pick(m_int, "auc_micro"),
                "internal_accuracy_hybridb": _pick(m_int, "accuracy"),
                "internal_balanced_accuracy_hybridb": _pick(m_int, "balanced_accuracy"),
                "external_auc_macro_hybridb": _pick(m_ext, "auc_macro"),
                "external_auc_micro_hybridb": _pick(m_ext, "auc_micro"),
                "external_accuracy_hybridb": _pick(m_ext, "accuracy"),
                "external_balanced_accuracy_hybridb": _pick(m_ext, "balanced_accuracy"),
                "has_external_merge_hybridb": int(m_ext is not None),
                "error": err_msg,
            }
        )

    summary_df = pd.DataFrame(rows)
    rank_col = str(args.rank_by)
    if rank_col not in summary_df.columns:
        _log(f"Ranking column not found: {rank_col}. Using internal_auc_macro_hybridb.")
        rank_col = "internal_auc_macro_hybridb"
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
