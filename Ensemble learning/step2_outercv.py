"""
Evaluate Hybrid-B with outer cross-validation and nested Stage 1 out-of-fold predictions.
Reuse clean_intersection.csv and meta.json from Step 1. Save internal OOF predictions, per-fold coefficients, external predictions, and the final external-inference model under 03_stage2. Write evaluation metrics under 04_metrics/hybridb_outercv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import joblib

from hybridb_common import (
    CLASS_ABBR,
    build_stage1_table_for_backbone,
    ensure_dir,
    is_external,
    is_internal,
    label_to_index,
    parse_mils_arg,
    safe_name,
    save_metrics_bundle,
    stage1_inner_oof_for_backbone,
    stage2_multinomial_lr,
    fit_predict_stage1_ovr,
)
from softmax_probs import CLASS_ORDER


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}][Step2] {msg}", flush=True)


def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = True):
    """Use tqdm when installed, with plain iteration and log messages as a fallback."""
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave)
    except Exception:
        return iterable


def _load_step1_inputs(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    clean_path = out_dir / "02_stage1" / "inputs" / "clean_intersection.csv"
    meta_path = out_dir / "02_stage1" / "meta.json"
    if not clean_path.exists():
        raise FileNotFoundError('Missing 02_stage1/inputs/clean_intersection.csv. Run step1_stage1_ovr.py first.')
    if not meta_path.exists():
        raise FileNotFoundError('Missing 02_stage1/meta.json. Run step1_stage1_ovr.py first.')
    df = pd.read_csv(clean_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=r"Ensemble_Results\3 models")
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--inner_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1_C", type=float, default=1)
    p.add_argument("--stage2_C", type=float, default=1)
    p.add_argument("--stage1_class_weight", type=str, default="balanced", help='Optional class weighting: balanced')
    p.add_argument("--mils", type=str, default="", help='Override the Step 1 MIL list with names present in the Step 0 and Step 1 features')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    clean, meta = _load_step1_inputs(out_dir)
    _log(f"Loaded the Step 1 strict intersection: {len(clean)} rows from {(out_dir / '02_stage1' / 'inputs' / 'clean_intersection.csv').as_posix()}）")

    # Use Step 1 MIL methods and backbones unless --mils selects a supported subset.
    mils_from_step1: List[str] = list(meta.get("mils", []))
    backbones: List[str] = list(meta.get("backbones", []))
    mils_override = parse_mils_arg(args.mils)
    mils = mils_from_step1 if mils_override is None else [m.strip().upper() for m in mils_override]
    if not mils or not backbones:
        raise ValueError('Step 1 meta.json must contain mils and backbones.')
    _log(
        f"mils={mils} | backbones={backbones} | outer_folds={args.outer_folds} | inner_folds={args.inner_folds} | "
        f"seed={args.seed} | stage1_C={args.stage1_C} | stage2_C={args.stage2_C}"
    )

    internal = clean[clean["dataset_type"].apply(is_internal)].copy()
    if "true_label" not in internal.columns:
        raise ValueError('INTERNAL requires true_label for Stage 2 training and evaluation.')
    externals = clean[clean["dataset_type"].apply(is_external)].copy()
    if len(externals) > 0:
        externals = externals.drop_duplicates(subset=["file_id", "dataset_type"]).copy()
    _log(f"Dataset split: internal={len(internal)} | external={len(externals)}")

    y_internal = label_to_index(internal["true_label"])
    from sklearn.model_selection import StratifiedKFold

    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    p_oof = np.zeros((len(internal), len(CLASS_ORDER)), dtype=np.float64)

    out_root = out_dir / "03_stage2"
    fold_root = out_root / "fold_artifacts"
    ensure_dir(out_root)
    ensure_dir(fold_root)

    cw = args.stage1_class_weight.strip() or None

    splits = list(outer.split(np.zeros(len(internal)), y_internal))
    _log('Starting outer CV. Stage 1 OOF predictions are recomputed within each outer fold.')
    for fold_idx, (tr_idx, va_idx) in enumerate(
        _progress(splits, desc="OuterCV(fold)", total=len(splits)), start=1
    ):
        t_fold0 = time.time()
        df_tr_all = internal.iloc[tr_idx].copy()
        df_va_all = internal.iloc[va_idx].copy()
        y_tr = y_internal[tr_idx]
        y_va = y_internal[va_idx]
        _log(f"fold={fold_idx}/{args.outer_folds}：train={len(df_tr_all)} | val={len(df_va_all)}")

        fold_dir = fold_root / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        # -------- Stage1 (nested) --------
        stage1_oof_parts = []
        stage1_val_parts = []
        stage1_coef_dump = {}

        for bb in _progress(backbones, desc=f"  fold{fold_idx}:Stage1(backbone)", total=len(backbones), leave=False):
            t0 = time.time()
            tbl_tr = build_stage1_table_for_backbone(df_tr_all, backbone=bb, mils=mils)
            tbl_va = build_stage1_table_for_backbone(df_va_all, backbone=bb, mils=mils)

            oof_bb = stage1_inner_oof_for_backbone(
                df_train=tbl_tr,
                backbone=bb,
                mils=mils,
                inner_folds=args.inner_folds,
                seed=args.seed,
                C=args.stage1_C,
                class_weight=cw,
            )
            pred_va_bb, models_full = fit_predict_stage1_ovr(
                df_train=tbl_tr,
                df_pred=tbl_va,
                backbone=bb,
                mils=mils,
                C=args.stage1_C,
                class_weight=cw,
            )

            stage1_oof_parts.append(oof_bb[["file_id"] + [f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER]])
            stage1_val_parts.append(pred_va_bb[["file_id"] + [f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER]])

            coef = {}
            for cls, lr in models_full.items():
                ab = CLASS_ABBR[cls]
                coef[cls] = {
                    "intercept": float(lr.intercept_[0]),
                    "coef": {f"{mil}_p_{ab}": float(w) for mil, w in zip(mils, lr.coef_[0])},
                }
            stage1_coef_dump[bb] = coef
            _log(f"fold={fold_idx} backbone={bb}: Stage 1 completed in {time.time() - t0:.1f}s")

        def _merge_parts(base: pd.DataFrame, parts: List[pd.DataFrame]) -> pd.DataFrame:
            z = base[["file_id"]].copy()
            for part in parts:
                z = pd.merge(z, part, on="file_id", how="inner")
            return z

        X_tr_df = _merge_parts(df_tr_all, stage1_oof_parts)
        X_va_df = _merge_parts(df_va_all, stage1_val_parts)

        feat_cols: List[str] = []
        for bb in backbones:
            feat_cols.extend([f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER])
        X_tr = X_tr_df[feat_cols].to_numpy(dtype=np.float32)
        X_va = X_va_df[feat_cols].to_numpy(dtype=np.float32)

        # -------- Stage2 --------
        _log(f"fold={fold_idx}: Fitting and validating Stage 2 multinomial logistic regression...")
        stage2 = stage2_multinomial_lr(C=args.stage2_C)
        stage2.fit(X_tr, y_tr)
        p_va = stage2.predict_proba(X_va)
        p_oof[va_idx] = p_va
        _log(f"fold={fold_idx}: Stage 2 completed in {time.time() - t_fold0:.1f}s including Stage 1")

        # artifacts
        with open(fold_dir / "stage1_coef.json", "w", encoding="utf-8") as f:
            json.dump(stage1_coef_dump, f, ensure_ascii=False, indent=2)

        lr2: LogisticRegression = stage2.named_steps["lr"]
        with open(fold_dir / "stage2_coef.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_cols": feat_cols,
                    "intercept": lr2.intercept_.tolist(),
                    "coef": lr2.coef_.tolist(),
                    "classes": list(map(int, lr2.classes_)),
                    "C": args.stage2_C,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        fold_pred = df_va_all[["file_id", "fold", "dataset_type", "true_label"]].copy()
        for i, cls in enumerate(CLASS_ORDER):
            fold_pred[f"{cls}_prob"] = p_va[:, i]
        fold_pred["pred_label"] = [CLASS_ORDER[i] for i in np.argmax(p_va, axis=1)]
        fold_pred["is_correct"] = fold_pred["pred_label"].astype(str) == fold_pred["true_label"].astype(str)
        fold_pred.to_csv(fold_dir / "val_predictions.csv", index=False)
        save_metrics_bundle(y_true=y_va, y_score=p_va, out_dir=fold_dir / "metrics", title_suffix=f"hybridb_outer_fold{fold_idx}")
        _log(f"fold={fold_idx}: Saved fold artifacts to {fold_dir.as_posix()}")

    # overall INTERNAL OOF
    _log('Outer CV complete. Aggregating INTERNAL OOF predictions and metrics...')
    internal_oof_pred = internal[["file_id", "fold", "dataset_type", "true_label"]].copy()
    for i, cls in enumerate(CLASS_ORDER):
        internal_oof_pred[f"{cls}_prob"] = p_oof[:, i]
    internal_oof_pred["pred_label"] = [CLASS_ORDER[i] for i in np.argmax(p_oof, axis=1)]
    internal_oof_pred["is_correct"] = internal_oof_pred["pred_label"].astype(str) == internal_oof_pred["true_label"].astype(str)
    internal_oof_pred.to_csv(out_root / "internal_oof_predictions.csv", index=False)
    save_metrics_bundle(
        y_true=y_internal,
        y_score=p_oof,
        out_dir=out_dir / "04_metrics" / "hybridb_outercv" / "INTERNAL_OOF",
        title_suffix="hybridb_outercv_INTERNAL_OOF",
    )
    _log(f"Saved INTERNAL OOF predictions: {(out_root / 'internal_oof_predictions.csv').as_posix()}")

    # -------- final model for external inference --------
    if len(externals) > 0:
        _log('Fitting the final model on all INTERNAL rows for EXTERNAL inference...')
        deploy_dir = out_root / "final_model_for_external"
        ensure_dir(deploy_dir)

        # stage1 OOF on full INTERNAL (for stage2 train)
        stage1_oof_parts_full = []
        stage1_models_full: Dict[str, Dict[str, LogisticRegression]] = {}
        for bb in _progress(backbones, desc="FinalFit:Stage1(backbone)", total=len(backbones)):
            t0 = time.time()
            tbl_int = build_stage1_table_for_backbone(internal, backbone=bb, mils=mils)
            oof_bb = stage1_inner_oof_for_backbone(
                df_train=tbl_int,
                backbone=bb,
                mils=mils,
                inner_folds=args.inner_folds,
                seed=args.seed,
                C=args.stage1_C,
                class_weight=cw,
            )
            stage1_oof_parts_full.append(oof_bb[["file_id"] + [f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER]])

            _, models_full = fit_predict_stage1_ovr(
                df_train=tbl_int,
                df_pred=tbl_int.iloc[:1].copy(),
                backbone=bb,
                mils=mils,
                C=args.stage1_C,
                class_weight=cw,
            )
            stage1_models_full[bb] = models_full
            _log(f"final backbone={bb}: Stage 1 full fit completed in {time.time() - t0:.1f}s")

        X_full_df = internal[["file_id"]].copy()
        for part in stage1_oof_parts_full:
            X_full_df = pd.merge(X_full_df, part, on="file_id", how="inner")
        feat_cols = []
        for bb in backbones:
            feat_cols.extend([f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER])
        X_full = X_full_df[feat_cols].to_numpy(dtype=np.float32)

        stage2_final = stage2_multinomial_lr(C=args.stage2_C)
        stage2_final.fit(X_full, y_internal)
        joblib.dump(
            {"mils": mils, "backbones": backbones, "stage1_models": stage1_models_full, "stage2": stage2_final},
            deploy_dir / "hybridb_final.joblib",
        )
        _log(f"Saved the final model: {(deploy_dir / 'hybridb_final.joblib').as_posix()}")

        groups = list(externals.groupby("dataset_type"))
        # Collect labeled external predictions for pooled metrics.
        merged_ext_preds: List[pd.DataFrame] = []
        merged_ext_scores: List[np.ndarray] = []
        for dt, g in _progress(groups, desc="External(dataset)", total=len(groups)):
            dt_name = safe_name(str(dt))
            ext_feat = g[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in g.columns else [])].copy()

            for bb in _progress(backbones, desc=f"  {dt_name}:backbone", total=len(backbones), leave=False):
                tbl_ext = build_stage1_table_for_backbone(g, backbone=bb, mils=mils)
                for cls in CLASS_ORDER:
                    ab = CLASS_ABBR[cls]
                    feat = [f"{mil}_p_{ab}" for mil in mils]
                    X = tbl_ext[feat].to_numpy(dtype=np.float32)
                    if cls in stage1_models_full[bb]:
                        ext_feat[f"{bb}_p_{ab}"] = stage1_models_full[bb][cls].predict_proba(X)[:, 1]
                    else:
                        prior = float((internal["true_label"].astype(str) == cls).mean())
                        ext_feat[f"{bb}_p_{ab}"] = prior

            X_ext = ext_feat[feat_cols].to_numpy(dtype=np.float32)
            p_ext = stage2_final.predict_proba(X_ext)
            ext_pred = ext_feat[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in ext_feat.columns else [])].copy()
            for i, cls in enumerate(CLASS_ORDER):
                ext_pred[f"{cls}_prob"] = p_ext[:, i]
            ext_pred["pred_label"] = [CLASS_ORDER[i] for i in np.argmax(p_ext, axis=1)]
            if "true_label" in ext_pred.columns:
                ext_pred["is_correct"] = ext_pred["pred_label"].astype(str) == ext_pred["true_label"].astype(str)
            ext_pred.to_csv(out_root / f"external_predictions__{dt_name}.csv", index=False)

            if "true_label" in ext_pred.columns and ext_pred["true_label"].notna().all():
                y_ext = label_to_index(ext_pred["true_label"])
                save_metrics_bundle(
                    y_true=y_ext,
                    y_score=p_ext,
                    out_dir=out_dir / "04_metrics" / "hybridb_outercv" / dt_name,
                    title_suffix=f"hybridb_final_{dt_name}",
                )
                merged_ext_preds.append(ext_pred)
                merged_ext_scores.append(p_ext)
            _log(f"Saved EXTERNAL predictions for dataset={dt_name} -> {(out_root / f'external_predictions__{dt_name}.csv').as_posix()}")

        # Compute pooled metrics using external sets with complete true labels.
        if merged_ext_preds and merged_ext_scores:
            ext_merge_pred = pd.concat(merged_ext_preds, axis=0, ignore_index=True)
            ext_merge_pred.to_csv(out_root / "external_predictions__EXTERNAL_MERGE.csv", index=False)

            p_merge = np.concatenate(merged_ext_scores, axis=0)
            y_merge = label_to_index(ext_merge_pred["true_label"])
            save_metrics_bundle(
                y_true=y_merge,
                y_score=p_merge,
                out_dir=out_dir / "04_metrics" / "hybridb_outercv" / "EXTERNAL_MERGE",
                title_suffix="hybridb_final_EXTERNAL_MERGE",
            )
            _log(f"Saved EXTERNAL_MERGE metrics and predictions: {(out_root / 'external_predictions__EXTERNAL_MERGE.csv').as_posix()}")
        else:
            _log('Skipping EXTERNAL_MERGE because no external set has complete true labels.')

    _log(f"done. output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()


