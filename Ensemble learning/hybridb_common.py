"""Shared utilities for the three Hybrid-B ensemble stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import metrics as metrics_mod
from softmax_probs import CLASS_ORDER, with_softmax_probs

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError('Install joblib to load or save ensemble models.') from e


CLASS_ABBR = {
    "Cholangiocarcinoma": "C",
    "Ampullary": "A",
    "Duodenal": "D",
    "Pancreatic": "P",
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("\\", "_")


def normalize_dataset_type(x: object) -> str:
    return str(x).strip().upper()


def is_internal(dt: object) -> bool:
    return normalize_dataset_type(dt) == "INTERNAL"


def is_train(dt: object) -> bool:
    return normalize_dataset_type(dt) == "TRAIN"


def is_external(dt: object) -> bool:
    d = normalize_dataset_type(dt)
    return d not in {"INTERNAL", "TRAIN"}


def label_to_index(y: pd.Series) -> np.ndarray:
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.number):
        y_int = arr.astype(np.int64)
        if y_int.min() == 1 and y_int.max() == len(CLASS_ORDER):
            y_int = y_int - 1
        return y_int
    mapping = {name: i for i, name in enumerate(CLASS_ORDER)}
    out = []
    for v in arr.astype(str):
        if v not in mapping:
            raise ValueError(f"Unknown true_label class: {v}. Expected one of {CLASS_ORDER}")
        out.append(mapping[v])
    return np.asarray(out, dtype=np.int64)


@dataclass(frozen=True)
class ModelRef:
    model_id: str  # MIL__Backbone
    mil: str
    backbone: str
    path: Path


def parse_mils_arg(mils_arg: str) -> Optional[List[str]]:
    mils_arg = (mils_arg or "").strip()
    if not mils_arg:
        return None
    parts = [p.strip() for p in mils_arg.split(",") if p.strip()]
    return parts or None


def discover_model_csvs(input_dir: Path, mils_filter: Optional[List[str]] = None) -> List[ModelRef]:
    csvs = sorted(input_dir.rglob("*.csv"))
    out: List[ModelRef] = []
    filt = {normalize_dataset_type(m) for m in mils_filter} if mils_filter else None
    for p in csvs:
        mil = normalize_dataset_type(p.parent.name)
        if filt is not None and mil not in filt:
            continue
        backbone = p.stem
        model_id = safe_name(f"{mil}__{backbone}")
        out.append(ModelRef(model_id=model_id, mil=mil, backbone=backbone, path=p))
    if not out:
        raise FileNotFoundError(f"No model CSVs found under {input_dir} after applying the MIL filter.")
    return out


def backbone_and_mil_from_model_id(model_id: str) -> Tuple[str, str]:
    parts = model_id.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model_id: {model_id}")
    return parts[1], parts[0]


def load_probability_csv(path: Path, model_id: str) -> pd.DataFrame:
    df = with_softmax_probs(pd.read_csv(path))
    df["dataset_type"] = df["dataset_type"].apply(normalize_dataset_type)
    # Use object dtype so external rows can carry a string fold label.
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype(object)
    # Different models can use different best-fold labels for external cases.
    # The fold column is part of the Step 1 join key.
    # Set external folds to "all" while preserving the internal OOF folds.
    df.loc[df["dataset_type"].apply(is_external), "fold"] = "all"
    key_cols = ["file_id", "fold", "dataset_type"]
    has_label = "true_label" in df.columns
    prob_cols = [f"{c}_prob" for c in CLASS_ORDER]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability column {missing}: {path}")
    cols = key_cols + (["true_label"] if has_label else []) + prob_cols
    sub = df[cols].copy()

    # Require a unique (file_id, fold, dataset_type) key to prevent duplicated join rows.
    # Keep true_label outside the key so missing external labels cannot remove rows.
    if sub.duplicated(subset=key_cols).any():
        agg_probs = sub.groupby(key_cols, as_index=False)[prob_cols].mean()
        if has_label:
            def _resolve_label(s: pd.Series):
                # Allow missing labels and reject conflicting nonempty labels.
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
    sub = sub.rename(columns={f"{c}_prob": f"{model_id}__{c}" for c in CLASS_ORDER})
    return sub


def merge_all_models(model_to_probs: Dict[str, Path]) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for model_id, p in sorted(model_to_probs.items(), key=lambda x: x[0]):
        df_m = load_probability_csv(p, model_id=model_id)
        if merged is None:
            merged = df_m
        else:
            key_cols = ["file_id", "fold", "dataset_type"]
            if "true_label" in merged.columns and "true_label" in df_m.columns:
                key_cols = key_cols + ["true_label"]
            merged = pd.merge(merged, df_m, on=key_cols, how="outer")
    if merged is None:
        raise RuntimeError('The merged table is empty because no probability data was loaded.')
    return merged


def get_prob_feature_cols(merged: pd.DataFrame) -> List[str]:
    cols = []
    for c in merged.columns:
        if c.count("__") == 2 and any(c.endswith("__" + cls) for cls in CLASS_ORDER):
            cols.append(c)
    return cols


def build_stage1_table_for_backbone(merged_df: pd.DataFrame, backbone: str, mils: List[str]) -> pd.DataFrame:
    df = merged_df
    out = df[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in df.columns else [])].copy()
    for mil in mils:
        model_id = f"{mil}__{backbone}"
        for cls in CLASS_ORDER:
            ab = CLASS_ABBR[cls]
            src = f"{model_id}__{cls}"
            dst = f"{mil}_p_{ab}"
            if src not in df.columns:
                raise ValueError(f"Missing feature column {src}. Check the Step 0 outputs and MIL filter.")
            out[dst] = df[src].astype(float)
    return out


def fit_predict_stage1_ovr(
    df_train: pd.DataFrame,
    df_pred: pd.DataFrame,
    backbone: str,
    mils: List[str],
    C: float = 1.0,
    class_weight: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, LogisticRegression]]:
    if "true_label" not in df_train.columns:
        raise ValueError('Stage 1 requires true_label and trains on INTERNAL rows.')
    y_full = df_train["true_label"].astype(str)

    pred_out = df_pred[["file_id", "fold", "dataset_type"] + (["true_label"] if "true_label" in df_pred.columns else [])].copy()
    models: Dict[str, LogisticRegression] = {}

    for cls in CLASS_ORDER:
        ab = CLASS_ABBR[cls]
        feat_cols = [f"{mil}_p_{ab}" for mil in mils]
        X_tr = df_train[feat_cols].to_numpy(dtype=np.float32)
        y_tr = (y_full == cls).astype(int).to_numpy()

        pos = int(y_tr.sum())
        neg = int(len(y_tr) - pos)
        if pos == 0 or neg == 0:
            prior = float(pos / max(len(y_tr), 1))
            pred_out[f"{backbone}_p_{ab}"] = prior
            continue

        lr = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,
        )
        lr.fit(X_tr, y_tr)
        models[cls] = lr

        X_pd = df_pred[feat_cols].to_numpy(dtype=np.float32)
        pred_out[f"{backbone}_p_{ab}"] = lr.predict_proba(X_pd)[:, 1]

    return pred_out, models


def stage1_inner_oof_for_backbone(
    df_train: pd.DataFrame,
    backbone: str,
    mils: List[str],
    inner_folds: int = 5,
    seed: int = 42,
    C: float = 1.0,
    class_weight: Optional[str] = None,
) -> pd.DataFrame:
    if "true_label" not in df_train.columns:
        raise ValueError('Stage 1 OOF requires true_label.')

    y_mc = label_to_index(df_train["true_label"])
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    oof = df_train[["file_id", "fold", "dataset_type", "true_label"]].copy()
    for cls in CLASS_ORDER:
        oof[f"{backbone}_p_{CLASS_ABBR[cls]}"] = np.nan

    for tr_idx, va_idx in skf.split(np.zeros(len(df_train)), y_mc):
        df_tr = df_train.iloc[tr_idx].copy()
        df_va = df_train.iloc[va_idx].copy()
        pred_va, _ = fit_predict_stage1_ovr(
            df_train=df_tr,
            df_pred=df_va,
            backbone=backbone,
            mils=mils,
            C=C,
            class_weight=class_weight,
        )
        for cls in CLASS_ORDER:
            ab = CLASS_ABBR[cls]
            oof.iloc[va_idx, oof.columns.get_loc(f"{backbone}_p_{ab}")] = pred_va[f"{backbone}_p_{ab}"].values

    for cls in CLASS_ORDER:
        ab = CLASS_ABBR[cls]
        col = f"{backbone}_p_{ab}"
        if oof[col].isna().any():
            prior = float((df_train["true_label"].astype(str) == cls).mean())
            oof[col] = oof[col].fillna(prior)
    return oof


def save_metrics_bundle(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, title_suffix: str) -> None:
    ensure_dir(out_dir)
    y_pred = np.argmax(y_score, axis=1)
    overall = metrics_mod.compute_overall_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))
    classwise = metrics_mod.compute_classwise_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))

    def _to_jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    with open(out_dir / "overall.json", "w", encoding="utf-8") as f:
        json.dump({k: _to_jsonable(v) for k, v in overall.items()}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "classwise.json", "w", encoding="utf-8") as f:
        json.dump(classwise, f, ensure_ascii=False, indent=2)

    metrics_mod.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(CLASS_ORDER),
        title_suffix=title_suffix,
        save_path=str(out_dir / "confusion_matrix.pdf"),
    )
    metrics_mod.plot_roc_curves(
        y_true=y_true,
        y_score=y_score,
        class_names=list(CLASS_ORDER),
        title_suffix=title_suffix,
        save_dir=str(out_dir),
    )


def stage2_multinomial_lr(C: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    C=C,
                    max_iter=5000,
                ),
            ),
        ]
    )


