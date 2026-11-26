"""
Fit a one-vs-rest MIL ensemble for each backbone using the Step 0 probability CSVs.
Save the outer-joined input, strict intersection, and internal and external subsets under 02_stage1/inputs. Write backbone OOF and external predictions and fitted models under 02_stage1/stage1_outputs. The full-internal OOF is for inspection. Step 2 uses the saved intersection and generates Stage 1 OOF predictions within each outer fold.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from hybridb_common import (
    CLASS_ABBR,
    backbone_and_mil_from_model_id,
    build_stage1_table_for_backbone,
    ensure_dir,
    get_prob_feature_cols,
    is_external,
    is_internal,
    merge_all_models,
    normalize_dataset_type,
    parse_mils_arg,
    safe_name,
    stage1_inner_oof_for_backbone,
    fit_predict_stage1_ovr,
)

import joblib
from softmax_probs import CLASS_ORDER


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}][Step1] {msg}", flush=True)


def _progress(iterable, *, desc: str, total: Optional[int] = None, leave: bool = True):
    """Use tqdm when installed, with plain iteration and log messages as a fallback."""
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=leave)
    except Exception:
        return iterable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default=r"Ensemble_Results\3 models")
    p.add_argument("--inner_folds", type=int, default=5) # Number of folds
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1)
    p.add_argument("--class_weight", type=str, default="", help='Optional class weighting: balanced')
    p.add_argument("--mils", type=str, default="", help='Comma-separated MIL names. Defaults to models found in Step 0.')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    probability_dir = out_dir / "01_softmax" / "probability_csv"
    if not probability_dir.exists():
        raise FileNotFoundError('Step 0 output is missing: 01_softmax/probability_csv')

    # Load the Step 0 probability outputs.
    model_to_probs = {p.stem: p for p in probability_dir.glob("*.csv")}
    if not model_to_probs:
        raise FileNotFoundError('No CSVs found in 01_softmax/probability_csv.')
    _log(f"Found Step 0 probability CSVs: {len(model_to_probs)} under {probability_dir}）")

    # Infer MIL methods and backbones, with an optional --mils override.
    all_model_ids = sorted(model_to_probs.keys())
    mils_auto = sorted({backbone_and_mil_from_model_id(mid)[1] for mid in all_model_ids})
    backbones = sorted({backbone_and_mil_from_model_id(mid)[0] for mid in all_model_ids})
    mils = parse_mils_arg(args.mils)
    if mils is None:
        mils = mils_auto
    else:
        mils = [normalize_dataset_type(m) for m in mils]
    _log(f"mils={mils} | backbones={backbones} | inner_folds={args.inner_folds} | seed={args.seed}")

    # Join all models and save their strict intersection for Step 2.
    inputs_dir = out_dir / "02_stage1" / "inputs"
    ensure_dir(inputs_dir)
    _log('Joining prediction tables and computing the strict intersection...')
    merged = merge_all_models(model_to_probs)
    merged.to_csv(inputs_dir / "merged_all.csv", index=False)

    prob_cols = get_prob_feature_cols(merged)
    clean = merged.dropna(subset=prob_cols).copy()
    clean.to_csv(inputs_dir / "clean_intersection.csv", index=False)

    internal = clean[clean["dataset_type"].apply(is_internal)].copy()
    externals = clean[clean["dataset_type"].apply(is_external)].copy()
    _log(
        f"Join complete. merged={len(merged)} rows | clean_intersection={len(clean)} rows | "
        f"internal={len(internal)} rows | external={len(externals)} rows"
    )
    internal.to_csv(inputs_dir / "internal_clean.csv", index=False)
    for dt, g in externals.groupby("dataset_type"):
        dt_name = safe_name(str(dt))
        g.to_csv(inputs_dir / f"external_clean__{dt_name}.csv", index=False)

    # Stage 1 output tables
    out1_dir = out_dir / "02_stage1" / "stage1_outputs"
    model_dir = out1_dir / "models"
    ensure_dir(out1_dir)
    ensure_dir(model_dir)
    # Save metadata for Step 2.
    meta = {
        "mils": mils,
        "backbones": backbones,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "C": args.C,
        "class_weight": args.class_weight.strip() or None,
    }
    with open(out_dir / "02_stage1" / "meta.json", "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, ensure_ascii=False, indent=2)

    cw = args.class_weight.strip() or None

    # Generate inner-CV OOF predictions for each backbone.
    internal_parts: List[pd.DataFrame] = []
    stage1_models_full: Dict[str, Dict[str, object]] = {}

    _log('Starting Stage 1 inner-CV OOF for INTERNAL cases...')
    for bb in _progress(backbones, desc="Stage1(backbone)", total=len(backbones)):
        t0 = time.time()
        _log(f"backbone={bb}: Starting inner-CV OOF and full-data fitting...")
        tbl_int = build_stage1_table_for_backbone(internal, backbone=bb, mils=mils)
        oof_bb = stage1_inner_oof_for_backbone(
            df_train=tbl_int,
            backbone=bb,
            mils=mils,
            inner_folds=args.inner_folds,
            seed=args.seed,
            C=args.C,
            class_weight=cw,
        )
        internal_parts.append(
            oof_bb[
                ["file_id", "fold", "dataset_type", "true_label"]
                + [f"{bb}_p_{CLASS_ABBR[c]}" for c in CLASS_ORDER]
            ]
        )

        # Fit all internal cases for reproducible external inference.
        _, models_full = fit_predict_stage1_ovr(
            df_train=tbl_int,
            df_pred=tbl_int.iloc[:1].copy(),
            backbone=bb,
            mils=mils,
            C=args.C,
            class_weight=cw,
        )
        stage1_models_full[bb] = {"backbone": bb, "mils": mils, "models": models_full}
        joblib.dump(stage1_models_full[bb], model_dir / f"{bb}_ovr_models.joblib")
        _log(f"backbone={bb}: Completed in {time.time() - t0:.1f}s")

    # Join backbone predictions on file_id.
    internal_oof = internal_parts[0]
    for part in internal_parts[1:]:
        internal_oof = pd.merge(
            internal_oof,
            part.drop(columns=["fold", "dataset_type", "true_label"]),
            on=["file_id"],
            how="inner",
        )
    internal_oof.to_csv(out1_dir / "internal_backbone_oof.csv", index=False)
    _log(f"Saved INTERNAL OOF predictions: {(out1_dir / 'internal_backbone_oof.csv').as_posix()}")

    # Use the full-data Stage 1 models for external inference.
    if len(externals) > 0:
        _log('Starting EXTERNAL inference with the fitted Stage 1 OvR models...')
        groups = list(externals.groupby("dataset_type"))
        for dt, g in _progress(groups, desc="External(dataset)", total=len(groups)):
            dt_name = safe_name(str(dt))
            preds_dt = g[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in g.columns else [])].copy()
            for bb in _progress(backbones, desc=f"  {dt_name}:backbone", total=len(backbones), leave=False):
                tbl_ext = build_stage1_table_for_backbone(g, backbone=bb, mils=mils)
                bundle = joblib.load(model_dir / f"{bb}_ovr_models.joblib")
                models_full = bundle["models"]
                for cls in CLASS_ORDER:
                    ab = CLASS_ABBR[cls]
                    feat_cols = [f"{mil}_p_{ab}" for mil in mils]
                    X = tbl_ext[feat_cols].to_numpy()
                    if cls in models_full:
                        preds_dt[f"{bb}_p_{ab}"] = models_full[cls].predict_proba(X)[:, 1]
                    else:
                        prior = float((internal["true_label"].astype(str) == cls).mean())
                        preds_dt[f"{bb}_p_{ab}"] = prior
            out_path = out1_dir / f"external_backbone__{dt_name}.csv"
            try:
                preds_dt.to_csv(out_path, index=False)
                _log(f"Saved EXTERNAL predictions for dataset={dt_name} -> {out_path.as_posix()}")
            except PermissionError:
                # Use an alternate filename if the CSV is locked, for example by Excel.
                alt = out1_dir / f"external_backbone__{dt_name}__{time.strftime('%Y%m%d_%H%M%S')}.csv"
                preds_dt.to_csv(alt, index=False)
                _log(f"Saved under an alternate filename because the output was locked. dataset={dt_name} -> {alt.as_posix()}")

    _log(f"done. output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()


