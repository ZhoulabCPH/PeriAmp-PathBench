#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge CellViT patch counts with cluster/IG/Attention and make summaries.

The CellViT input is the ``*_patch_cell_counts.csv`` written by
``cellvit/cellvit_patch_assign.py``.  The join key is the project's patch
coordinate system: ``(case_id, coord_x, coord_y)`` ==
``(sample_id, patch_tile_x, patch_tile_y)``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# =====================================================================
# Default settings
# =====================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    r"outputs/interpretability"
)

# Inputs to export_patch_table.py
DEFAULT_CLUSTER_DUMP = DEFAULT_DATA_ROOT / "tumor_cluster"
DEFAULT_IG_NPZ = DEFAULT_DATA_ROOT / "IG_output"
DEFAULT_ATTN_NPZ = DEFAULT_DATA_ROOT / "attention_output" / "attentions"
DEFAULT_ANNOTATION_DIR = DEFAULT_DATA_ROOT / "tumor_area_output"

# CellViT tile counts, cell assignments, and morphology data
DEFAULT_CELL_COUNTS_DIR = DEFAULT_DATA_ROOT / "cell_patch_assignment"
DEFAULT_ASSIGNMENT_DIR = DEFAULT_DATA_ROOT / "cell_patch_assignment"
DEFAULT_MORPHOLOGY_DIR = DEFAULT_DATA_ROOT / "tumor_cell_morphology"

# Intermediate and final outputs
DEFAULT_ANALYSIS_ROOT = SCRIPT_DIR / "integrated_cellvit_analysis"
DEFAULT_PATCH_DIR = DEFAULT_ANALYSIS_ROOT / "patch_table"
DEFAULT_CLUSTER_SUMMARY = DEFAULT_ANALYSIS_ROOT / "cluster_summary.csv"
DEFAULT_OUT_DIR = DEFAULT_ANALYSIS_ROOT / "final"

# Rebuild patch tables from clustering, IG, and attention when enabled.
# Reuse existing patch tables when rebuilding is disabled.
DEFAULT_REBUILD_PATCH_TABLE = True
DEFAULT_TILE_SIZE = 256
DEFAULT_MPP = None  # None infers MPP from the per-cell mpp_um_per_px field.
DEFAULT_ATTENTION_QUANTILE = 0.90
DEFAULT_IG_QUANTILE = 0.90

# =====================================================================

KEY = ["case_id", "coord_x", "coord_y"]
MORPH_FEATURES = [
    "area_um2", "perimeter_um", "major_axis_um", "minor_axis_um",
    "aspect_ratio", "eccentricity", "solidity", "circularity", "type_prob",
    "mpp_um_per_px",
]
TYPE_MORPH_FEATURES = ["area_um2", "eccentricity", "solidity", "circularity"]


def rebuild_patch_tables(args: argparse.Namespace) -> None:
    """Build the stage-1 patch table in-process.

    The implementation reuses the exporter module's tested loading/alignment
    functions, but does not start a second Python process or require a second
    command from the user.
    """
    from export_patch_table import (
        IG_VALUE_KEYS,
        ATTN_VALUE_KEYS,
        align_values,
        build_in_tumor_flags,
        collect_value_npz,
        load_cluster_dump,
        resolve_dump,
        write_cluster_summary,
        write_patch_table_by_case,
    )

    args.patch_dir.mkdir(parents=True, exist_ok=True)
    print('[1/2] Rebuilding patch tables from clustering, IG, and attention')
    dump_npz = resolve_dump(args.cluster_dump, None)
    dump = load_cluster_dump(dump_npz)
    coords_all = dump["coords"]
    case_ids_all = dump["case_ids"]
    cluster_id = dump["cluster_id"]
    feature_label = dump["meta"].get("feature_label") or None

    ig_records = collect_value_npz(
        args.ig_npz, IG_VALUE_KEYS, feature_label=feature_label,
        mil=None, fold=None, prefer_foldavg=False, tag="IG",
    )
    ig, ig_folds = align_values(ig_records, coords_all, case_ids_all, "IG")
    attn_records = collect_value_npz(
        args.attn_npz, ATTN_VALUE_KEYS, feature_label=feature_label,
        mil=None, fold=None, prefer_foldavg=True, tag="Attention",
    )
    attn, attn_folds = align_values(
        attn_records, coords_all, case_ids_all, "Attention"
    )
    in_tumor = build_in_tumor_flags(
        coords_all, case_ids_all, annotation_dir=args.annotation_dir, require=False
    )
    write_patch_table_by_case(
        out_dir=args.patch_dir, case_ids_all=case_ids_all,
        coords_all=coords_all, cluster_id=cluster_id, in_tumor=in_tumor,
        ig=ig, ig_folds=ig_folds, attn=attn, attn_folds=attn_folds,
        patch_root=None,
    )
    write_cluster_summary(
        args.cluster_summary, cluster_id, in_tumor, ig, attn
    )


def case_stem(value: object) -> str:
    text = Path(str(value)).stem
    for suffix in ("_cells", "_cell_detection", "_patch_cell_counts", "_patch_table"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    return text


def read_patch_tables(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*_patch_table.csv"))
    if not files:
        raise FileNotFoundError(f"No *_patch_table.csv files found: {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        required = {"case_id", "coord_x", "coord_y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing fields: {sorted(missing)}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return normalize_keys(out, "patch table")


def read_cell_counts(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*_patch_cell_counts.csv"))
    if not files:
        raise FileNotFoundError(f"No *_patch_cell_counts.csv files found: {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        rename = {
            "sample_id": "case_id", "patch_tile_x": "coord_x", "patch_tile_y": "coord_y"
        }
        df = df.rename(columns=rename)
        missing = set(KEY) - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing CellViT fields: {sorted(missing)}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = normalize_keys(out, "CellViT counts")
    dup = out.duplicated(KEY, keep=False)
    if dup.any():
        examples = out.loc[dup, KEY].head(5).to_dict("records")
        raise ValueError('CellViT counts include duplicate patch keys. Select one cell_source. Example: ' + str(examples))
    return out


def safe_token(value: object) -> str:
    raw = str(value).strip().lower()
    if raw.startswith("-"):
        raw = "minus_" + raw[1:]
    token = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_")
    return token or "unknown"


def _flatten_columns(columns: Iterable[object]) -> list[str]:
    result = []
    for col in columns:
        if isinstance(col, tuple):
            result.append("_".join(str(x) for x in col if str(x)))
        else:
            result.append(str(col))
    return result


def aggregate_morphology(
    morphology_dir: Path, assignment_dir: Path, case_ids: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate single-cell morphology to project patch coordinates."""
    outputs = []
    qc_rows = []
    morph_use = [
        "cell_index", "cell_type_name", "centroid_x_px", "centroid_y_px",
        "valid_geometry", "morphology_cluster", *MORPH_FEATURES,
    ]
    assign_use = [
        "cell_id", "cell_type", "in_tumor_patch", "patch_tile_x", "patch_tile_y",
        "centroid_x_px", "centroid_y_px",
    ]
    for case_id in sorted(set(map(str, case_ids))):
        morph_path = morphology_dir / case_id / "nuclear_morphology.csv"
        assign_path = assignment_dir / f"{case_id}_cells_cell_patch_assignment.csv"
        if not morph_path.is_file() or not assign_path.is_file():
            qc_rows.append({"case_id": case_id, "status": "missing_file",
                            "morphology_file": morph_path.is_file(),
                            "assignment_file": assign_path.is_file()})
            continue
        assignment = pd.read_csv(assign_path, usecols=assign_use)
        assignment = assignment[pd.to_numeric(assignment["in_tumor_patch"], errors="coerce").eq(1)].copy()
        assignment = assignment.dropna(subset=["centroid_x_px", "centroid_y_px", "patch_tile_x", "patch_tile_y"])
        assignment = assignment.rename(columns={
            "centroid_x_px": "assign_centroid_x_px",
            "centroid_y_px": "assign_centroid_y_px",
            "patch_tile_x": "coord_x", "patch_tile_y": "coord_y",
        })
        morphology = pd.read_csv(morph_path, usecols=lambda c: c in set(morph_use))
        missing = set(morph_use) - set(morphology.columns)
        if missing:
            raise ValueError(f"{morph_path} is missing fields: {sorted(missing)}")
        # Tumor filtering renumbers some cell_id values, so the stable key is
        # the CellViT centroid (stored to four decimals) plus cell type.
        for frame, x_col, y_col, type_col in (
            (assignment, "assign_centroid_x_px", "assign_centroid_y_px", "cell_type"),
            (morphology, "centroid_x_px", "centroid_y_px", "cell_type_name"),
        ):
            frame["_cx_key"] = (pd.to_numeric(frame[x_col], errors="coerce") * 10000).round().astype("Int64")
            frame["_cy_key"] = (pd.to_numeric(frame[y_col], errors="coerce") * 10000).round().astype("Int64")
            frame["_type_key"] = frame[type_col].astype(str).str.strip().str.lower()
        join_key = ["_cx_key", "_cy_key", "_type_key"]
        if assignment.duplicated(join_key).any() or morphology.duplicated(join_key).any():
            raise ValueError(f"{case_id}: Duplicate centroid and cell-type keys prevent a one-to-one join.")
        joined = assignment.merge(morphology, on=join_key, how="inner", validate="one_to_one")
        dx = (joined["assign_centroid_x_px"] - joined["centroid_x_px"]).abs()
        dy = (joined["assign_centroid_y_px"] - joined["centroid_y_px"]).abs()
        coord_ok = (dx.le(1e-4) & dy.le(1e-4))
        if len(joined) and float(coord_ok.mean()) < .999:
            raise ValueError(f"{case_id}: Cell coordinates differ too often between morphology and assignment: {coord_ok.mean():.2%}")
        joined = joined[coord_ok].copy()
        joined["coord_x"] = joined["coord_x"].astype("int64")
        joined["coord_y"] = joined["coord_y"].astype("int64")
        group_key = ["coord_x", "coord_y"]
        for col in MORPH_FEATURES:
            joined[col] = pd.to_numeric(joined[col], errors="coerce")
        basic = joined.groupby(group_key).agg(
            n_morphology_cells=("cell_index", "size"),
            n_valid_morphology=("valid_geometry", "sum"),
            **{f"morph_mean_{f}": (f, "mean") for f in MORPH_FEATURES},
            **{f"morph_std_{f}": (f, "std") for f in MORPH_FEATURES},
        )
        # Morphology phenotype counts/fractions.
        mc = pd.crosstab([joined["coord_x"], joined["coord_y"]], joined["morphology_cluster"])
        mc.columns = [f"n_morphology_cluster_{safe_token(c)}" for c in mc.columns]
        basic = basic.join(mc, how="left")
        for col in mc.columns:
            basic[col.replace("n_", "frac_", 1)] = basic[col] / basic["n_morphology_cells"]
        # Selected morphology means separately for every CellViT cell type.
        typed = (joined.groupby(group_key + ["cell_type_name"])[TYPE_MORPH_FEATURES]
                 .mean().unstack("cell_type_name"))
        typed.columns = [f"morph_mean_{safe_token(cell_type)}_{feature}"
                         for feature, cell_type in typed.columns]
        basic = basic.join(typed, how="left").reset_index()
        basic.insert(0, "case_id", case_id)
        outputs.append(basic)
        qc_rows.append({
            "case_id": case_id, "status": "ok", "morphology_file": True,
            "assignment_file": True, "assigned_cells": len(assignment),
            "joined_cells": len(joined), "cell_join_rate": len(joined) / len(assignment) if len(assignment) else np.nan,
            "coordinate_match_rate": float(coord_ok.mean()) if len(coord_ok) else np.nan,
            "patches_with_morphology": len(basic),
        })
        print(f"[morphology] {case_id}: cell={len(joined):,}, patch={len(basic):,}")
    if not outputs:
        raise RuntimeError('No cases have both per-cell morphology and assignment data.')
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(qc_rows)


def normalize_keys(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["case_id"] = out["case_id"].map(case_stem)
    for col in ("coord_x", "coord_y"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[KEY].isna().any().any():
        raise ValueError(f"{label} contains missing join keys.")
    out["coord_x"] = out["coord_x"].round().astype("int64")
    out["coord_y"] = out["coord_y"].round().astype("int64")
    return out


def add_cell_features(df: pd.DataFrame, tile_size: int, mpp: float | None) -> pd.DataFrame:
    out = df.copy()
    # Only these n_* columns are CellViT cell-type counts.  In particular,
    # never treat n_morphology_cluster_* or n_morphology_cells as CellViT
    # counts: their fractions were already computed against
    # n_morphology_cells during morphology aggregation.
    count_cols = [c for c in out.columns if re.fullmatch(r"n_.+", str(c))]
    count_cols = [
        c for c in count_cols
        if c not in {"n_cells", "n_ig_folds", "n_attn_folds"}
        and not str(c).startswith("n_morphology_")
        and c != "n_valid_morphology"
    ]
    for c in count_cols + (["n_cells"] if "n_cells" in out else []):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "n_cells" not in out:
        out["n_cells"] = out[count_cols].sum(axis=1) if count_cols else 0
    denom = out["n_cells"].replace(0, np.nan)
    for c in count_cols:
        fraction_col = c.replace("n_", "frac_", 1)
        # Preserve fractions supplied by morphology aggregation or another
        # upstream table; this prevents accidental denominator changes.
        if fraction_col not in out.columns:
            out[fraction_col] = out[c] / denom
    if mpp is not None:
        area_mm2 = (float(tile_size) * float(mpp) / 1000.0) ** 2
        out["patch_area_mm2"] = area_mm2
        out["cell_density_per_mm2"] = out["n_cells"] / area_mm2
    elif "morph_mean_mpp_um_per_px" in out:
        out["patch_area_mm2"] = (float(tile_size) * out["morph_mean_mpp_um_per_px"] / 1000.0) ** 2
        out["cell_density_per_mm2"] = out["n_cells"] / out["patch_area_mm2"]
    else:
        out["cell_density_per_mm2"] = np.nan
    return out


def add_evidence(df: pd.DataFrame, attention_q: float, ig_q: float) -> pd.DataFrame:
    out = df.copy()
    if "attention" not in out.columns and "ig" not in out.columns:
        out["joint_evidence_state"] = "no_model_evidence"
        return out
    for c in ("attention", "ig"):
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["abs_ig"] = out["ig"].abs()
    # Percentiles are computed within case because attention is bag-relative.
    out["attention_percentile"] = out.groupby("case_id")["attention"].rank(pct=True)
    out["abs_ig_percentile"] = out.groupby("case_id")["abs_ig"].rank(pct=True)
    out["high_attention"] = out["attention_percentile"] >= attention_q
    out["strong_ig"] = out["abs_ig_percentile"] >= ig_q
    out["positive_ig"] = out["ig"] > 0
    out["negative_ig"] = out["ig"] < 0
    conditions = [
        out.high_attention & out.strong_ig & out.positive_ig,
        out.high_attention & out.strong_ig & out.negative_ig,
        out.high_attention & ~out.strong_ig,
        ~out.high_attention & out.strong_ig & out.positive_ig,
        ~out.high_attention & out.strong_ig & out.negative_ig,
    ]
    labels = ["attention_supportive", "attention_contradictory", "attention_only",
              "ig_supportive_low_attention", "ig_contradictory_low_attention"]
    out["joint_evidence_state"] = np.select(conditions, labels, default="low_evidence")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Combine CellViT morphology, clustering, integrated gradients, and attention.')
    ap.add_argument("--cluster-dump", type=Path, default=DEFAULT_CLUSTER_DUMP,
                    help='Cohort dump from cluster_features.py')
    ap.add_argument("--ig-npz", type=Path, default=DEFAULT_IG_NPZ)
    ap.add_argument("--attn-npz", type=Path, default=DEFAULT_ATTN_NPZ)
    ap.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    ap.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    ap.add_argument("--cluster-summary", type=Path, default=DEFAULT_CLUSTER_SUMMARY)
    ap.add_argument("--cell-counts-dir", type=Path, default=DEFAULT_CELL_COUNTS_DIR)
    ap.add_argument("--morphology-dir", type=Path, default=DEFAULT_MORPHOLOGY_DIR,
                    help='Directory containing <case>/nuclear_morphology.csv')
    ap.add_argument("--assignment-dir", type=Path, default=DEFAULT_ASSIGNMENT_DIR,
                    help='Directory containing *_cell_patch_assignment.csv files')
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    ap.add_argument("--mpp", type=float, default=DEFAULT_MPP,
                    help='Microns per level-0 pixel. Defaults to the morphology metadata.')
    ap.add_argument("--attention-quantile", type=float, default=DEFAULT_ATTENTION_QUANTILE)
    ap.add_argument("--ig-quantile", type=float, default=DEFAULT_IG_QUANTILE)
    ap.add_argument("--rebuild-patch-table", dest="rebuild_patch_table",
                    action="store_true", default=DEFAULT_REBUILD_PATCH_TABLE,
                    help='Rebuild patch tables from clustering, IG, and attention. Enabled by default.')
    ap.add_argument("--no-rebuild-patch-table", dest="rebuild_patch_table",
                    action="store_false", help='Reuse existing patch tables')
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.attention_quantile <= 1 and 0 < args.ig_quantile <= 1):
        raise ValueError('quantile must be in (0, 1].')
    if args.tile_size <= 0:
        raise ValueError('--tile-size must be a positive integer.')
    patch_exists = any(args.patch_dir.glob("*_patch_table.csv"))
    if args.rebuild_patch_table or not patch_exists:
        rebuild_patch_tables(args)
    else:
        print(f"[1/2] Using existing patch tables: {args.patch_dir}")
    print('[2/2] Joining cell composition, morphology, clusters, IG, and attention')
    patch = read_patch_tables(args.patch_dir)
    cells = read_cell_counts(args.cell_counts_dir)
    morphology, morphology_qc = aggregate_morphology(
        args.morphology_dir,
        args.assignment_dir,
        patch["case_id"].unique(),
    )
    merged = patch.merge(cells, on=KEY, how="left", suffixes=("", "_cellvit"), indicator=True)
    merged["cellvit_match"] = merged["_merge"].eq("both")
    qc = pd.DataFrame([{
        "patch_rows": len(patch), "cellvit_rows": len(cells),
        "matched_rows": int(merged.cellvit_match.sum()),
        "unmatched_patch_rows": int((~merged.cellvit_match).sum()),
        "match_rate": float(merged.cellvit_match.mean()),
    }])
    merged = merged.drop(columns=["_merge"])
    merged = merged.merge(morphology, on=KEY, how="left", validate="one_to_one")
    merged["morphology_match"] = merged["n_morphology_cells"].notna()
    morph_expected = merged["n_cells"].fillna(0).gt(0)
    qc["patches_with_cells"] = int(morph_expected.sum())
    qc["patches_with_morphology"] = int(merged["morphology_match"].sum())
    qc["morphology_match_rate_among_nonempty"] = (
        float(merged.loc[morph_expected, "morphology_match"].mean())
        if morph_expected.any() else np.nan
    )
    merged = add_cell_features(merged, args.tile_size, args.mpp)
    merged = add_evidence(merged, args.attention_quantile, args.ig_quantile)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "integrated_patch_table.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(out / "merge_qc.csv", index=False, encoding="utf-8-sig")
    morphology_qc.to_csv(out / "morphology_merge_qc_by_case.csv", index=False, encoding="utf-8-sig")
    # NaN-aware summaries; each row is a patch, while case summary is suitable
    # for downstream case-level statistical testing.
    agg = {
        "n_patches": ("cluster", "size"), "n_cases": ("case_id", "nunique"),
        "n_cells": ("n_cells", "sum"), "mean_attention": ("attention", "mean"),
        "mean_ig": ("ig", "mean"), "mean_abs_ig": ("abs_ig", "mean"),
        "mean_cell_density": ("cell_density_per_mm2", "mean"),
    }
    for col in [c for c in merged.columns if c.startswith("n_") and c not in {"n_cells", "n_ig_folds", "n_attn_folds"}]:
        agg[f"{col}_sum"] = (col, "sum")
    for col in [c for c in merged.columns if c.startswith("frac_")]:
        agg[f"mean_{col}"] = (col, "mean")
    for col in [c for c in merged.columns if c.startswith("morph_mean_")]:
        agg[f"mean_{col}"] = (col, "mean")
    cluster = merged.groupby("cluster", dropna=False).agg(**agg).reset_index()
    case_agg = {
        "n_patches": ("cluster", "size"), "n_cells": ("n_cells", "sum"),
        "mean_attention": ("attention", "mean"), "mean_ig": ("ig", "mean"),
        "mean_abs_ig": ("abs_ig", "mean"),
    }
    for col in [c for c in merged.columns if c.startswith("n_") and c not in {"n_cells", "n_ig_folds", "n_attn_folds"}]:
        case_agg[f"{col}_sum"] = (col, "sum")
    for col in [c for c in merged.columns if c.startswith("morph_mean_")]:
        case_agg[f"mean_{col}"] = (col, "mean")
    case_cluster = merged.groupby(["case_id", "cluster"], dropna=False).agg(**case_agg).reset_index()
    state = (merged.groupby(["cluster", "joint_evidence_state"], dropna=False)
             .size().rename("n_patches").reset_index())
    cluster.to_csv(out / "cluster_cellvit_evidence_summary.csv", index=False, encoding="utf-8-sig")
    case_cluster.to_csv(out / "case_cluster_summary.csv", index=False, encoding="utf-8-sig")
    state.to_csv(out / "cluster_evidence_states.csv", index=False, encoding="utf-8-sig")
    morph_rate = (merged.loc[morph_expected, "morphology_match"].mean()
                  if morph_expected.any() else np.nan)
    print(f"[complete] patches={len(merged):,}, CellViT match rate={qc.match_rate.iloc[0]:.2%}, "
          f"Morphology match rate for patches with cells={morph_rate:.2%}")
    print(f"[output] {out.resolve()}")


if __name__ == "__main__":
    main()
