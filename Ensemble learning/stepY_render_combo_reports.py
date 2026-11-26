"""
Render metric tables and PDF plots for a saved ensemble run.
Select the run by --run_id under --ablation_root. Use --methods and --datasets to choose saved results. Use --all_external to render external datasets with complete labels. The script writes confusion matrices, ROC plots, and overall and per-class metric tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import metrics as metrics_mod
from hybridb_common import ensure_dir, label_to_index, safe_name, save_metrics_bundle
from softmax_probs import CLASS_ORDER


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}][Render] {msg}", flush=True)


def _read_summary(summary_path: Path) -> Optional[pd.DataFrame]:
    if not summary_path.exists():
        return None
    # Try legacy encodings used by summary CSVs on Windows.
    # Only the run ID and output directory are needed.
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(summary_path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return None


def _find_combo_dir_by_run_id(ablation_root: Path, run_id: str) -> Tuple[Path, Optional[Dict]]:
    """Return the combination directory and optional summary row. Prefer the output-directory column in summary_human.csv, then search 03_ablation_ensemble for the run ID."""
    summary_path = ablation_root / "summary_human.csv"
    df = _read_summary(summary_path)
    if df is not None and "run_id" in df.columns:
        hit = df[df["run_id"].astype(str) == str(run_id)].copy()
        if len(hit) > 0:
            row = hit.iloc[0].to_dict()
            # Some legacy summary CSVs have localized or misdecoded headers.
            # Look for the path value containing 03_ablation_ensemble.
            out_dir = ""
            if isinstance(row.get("输出目录", ""), str):
                out_dir = str(row.get("输出目录", "")).strip()
            if not out_dir:
                for _, v in row.items():
                    if not isinstance(v, str):
                        continue
                    s = v.strip()
                    s2 = s.replace("/", "\\")
                    if "03_ablation_ensemble" in s2 and "combo_k" in s2:
                        out_dir = s
                        break

            if out_dir:
                p_raw = Path(str(out_dir))

                # Use an absolute path directly.
                if p_raw.is_absolute():
                    return p_raw.resolve(), row

                # Resolve project-relative summary paths against the current directory.
                # Example: outputs/ensemble/hybridb_combo/03_ablation_ensemble/combos/combo_k3_example
                cwd = Path.cwd()
                cand_cwd = (cwd / p_raw).resolve()
                if cand_cwd.exists():
                    return cand_cwd, row

                # Also try paths relative to ablation_root.
                cand_root = (ablation_root / p_raw).resolve()
                if cand_root.exists():
                    return cand_root, row

                # Avoid duplicating ablation_root if the summary path already includes it.
                # Allow the stored relative path to begin with the ablation root directory.
                #     ablation_root="Ensemble_Results\_grid_test\all models combo"）
                try:
                    ar_norm = str(ablation_root).replace("/", "\\").lower().rstrip("\\")
                    pr_norm = str(p_raw).replace("/", "\\").lower()
                    if pr_norm.startswith(ar_norm):
                        cand = (cwd / p_raw).resolve()
                        if cand.exists():
                            return cand, row
                except Exception:
                    pass

                # Return the most likely path so an error can identify it.
                return cand_root, row

    # fallback search
    root_03 = ablation_root / "03_ablation_ensemble"
    if not root_03.exists():
        raise FileNotFoundError(f"Directory not found: {root_03.as_posix()}. Check that --ablation_root points to the ensemble output directory.")

    # Expected combination directory pattern
    pattern = f"combo_k*_{run_id}"
    matches = sorted([p for p in root_03.rglob(pattern) if p.is_dir()])
    if not matches:
        raise FileNotFoundError(
            f"No combination directory found for run_id={run_id}. Try the following:\n"
            f"  - Check whether {summary_path.as_posix()} exists and contains the run ID.\n"
            f"  - Or look under {root_03.as_posix()} for combos/combo_k*_{run_id}\n"
        )
    if len(matches) > 1:
        _log(f"Multiple directories matched. Using the first: {[m.as_posix() for m in matches[:5]]}{' ...' if len(matches) > 5 else ''}")
    return matches[0], None


def _rel_tag_dir_from_combo_dir(ablation_root: Path, combo_dir: Path) -> Path:
    """Return the path to the combination relative to 03_ablation_ensemble."""
    combo_dir = combo_dir.resolve()
    root_03 = (ablation_root / "03_ablation_ensemble").resolve()
    try:
        return combo_dir.relative_to(root_03)
    except Exception:
        parts = list(combo_dir.parts)
        key = "03_ablation_ensemble"
        if key in parts:
            idx = parts.index(key)
            rel = Path(*parts[idx + 1 :])
            if str(rel).strip():
                return rel
        # Fall back to the final path component.
        return Path(combo_dir.name)


def _parse_list_arg(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def _load_pred_csv(pred_csv: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray], np.ndarray]:
    df = pd.read_csv(pred_csv)
    prob_cols = [f"{c}_prob" for c in CLASS_ORDER]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{pred_csv.as_posix()} is missing probability columns: {missing}")
    y_score = df[prob_cols].to_numpy(dtype=np.float64)
    y_true = None
    if "true_label" in df.columns and df["true_label"].notna().all():
        y_true = label_to_index(df["true_label"])
    return df, y_true, y_score


def _dump_metrics_tables(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    try_xlsx: bool = True,
) -> None:
    ensure_dir(out_dir)
    y_pred = np.argmax(y_score, axis=1)
    overall = metrics_mod.compute_overall_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))
    classwise = metrics_mod.compute_classwise_metrics(y_true, y_pred, y_score, list(CLASS_ORDER))

    overall_table = pd.DataFrame([overall]).copy()
    # Exclude large prediction arrays from the metric tables.
    for col in ["true", "pred"]:
        if col in overall_table.columns:
            overall_table = overall_table.drop(columns=[col])

    classwise_table = pd.DataFrame(classwise).T.reset_index().rename(columns={"index": "class"})

    pfx = f"{prefix}_" if prefix else ""
    overall_table.to_csv(out_dir / f"{pfx}overall_table.csv", index=False, encoding="utf-8-sig")
    classwise_table.to_csv(out_dir / f"{pfx}classwise_table.csv", index=False, encoding="utf-8-sig")

    if not try_xlsx:
        return

    # Skip the optional XLSX export if openpyxl is unavailable.
    xlsx_path = out_dir / f"{pfx}metrics_tables.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            overall_table.to_excel(w, sheet_name="overall", index=False)
            classwise_table.to_excel(w, sheet_name="classwise", index=False)
    except Exception as e:
        _log(f"Skipping XLSX output: {e}")


def _render_one_dataset(
    *,
    pred_csv: Path,
    metrics_dir: Path,
    title_suffix: str,
    overwrite: bool,
) -> bool:
    """Render the dataset when its true labels are complete and report whether output was produced."""
    if not pred_csv.exists():
        _log(f"Skipping missing prediction file: {pred_csv.as_posix()}")
        return False

    df, y_true, y_score = _load_pred_csv(pred_csv)
    if y_true is None:
        _log(f"Skipping {pred_csv.name} because true_label values are incomplete.")
        return False

    ensure_dir(metrics_dir)

    # Reuse existing plots unless overwriting is requested, while filling missing outputs.
    cm_pdf = metrics_dir / "confusion_matrix.pdf"
    need_plot = overwrite or (not cm_pdf.exists())

    if need_plot:
        save_metrics_bundle(y_true=y_true, y_score=y_score, out_dir=metrics_dir, title_suffix=title_suffix)
        _log(f"Wrote PDF plots and JSON metrics: {metrics_dir.as_posix()}")
    else:
        # Generate missing JSON metrics.
        if not (metrics_dir / "overall.json").exists() or not (metrics_dir / "classwise.json").exists():
            save_metrics_bundle(y_true=y_true, y_score=y_score, out_dir=metrics_dir, title_suffix=title_suffix)
            _log(f"Generated missing JSON and PDF outputs: {metrics_dir.as_posix()}")
        else:
            _log(f"Reusing existing PDF and JSON outputs: {metrics_dir.as_posix()}")

    _dump_metrics_tables(y_true=y_true, y_score=y_score, out_dir=metrics_dir, prefix="", try_xlsx=True)

    # Record the source predictions alongside the report.
    meta = {
        "pred_csv": pred_csv.as_posix(),
        "n_samples": int(len(df)),
        "title_suffix": title_suffix,
        "generated_at": _now_str(),
    }
    with open(metrics_dir / "render_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_root", type=str, default=r"outputs/ensemble/hybridb_combo", help='Ensemble output root containing summary_human.csv and the predictions')
    p.add_argument("--run_id", type=str, default="98ff61d0", help='Eight-character run ID from summary_human.csv')
    p.add_argument("--methods", type=str, default="mean,vote,hybridb",
                   help='Comma-separated methods to render: mean, vote, hybridb')
    p.add_argument(
        "--datasets",
        type=str,
        default="INTERNAL_OOF,EXTERNAL_MERGE,EXTERNAL_SET_1,EXTERNAL_SET_2",
        help='Comma-separated datasets, including INTERNAL_OOF, EXTERNAL_MERGE, or an external dataset name',
    )
    p.add_argument("--all_external", action="store_true", help='Render all external prediction CSVs with complete labels')
    p.add_argument("--overwrite", default=True, action="store_true", help='Regenerate existing PDF plots')
    args = p.parse_args()

    ablation_root = Path(args.ablation_root)
    run_id = str(args.run_id).strip()
    if not run_id:
        raise ValueError('Provide a nonempty --run_id.')

    methods = [m.lower().strip() for m in _parse_list_arg(args.methods)]
    methods = [m for m in methods if m in {"mean", "vote", "hybridb"}]
    if not methods:
        raise ValueError('--methods must contain mean, vote, or hybridb.')

    combo_dir, row = _find_combo_dir_by_run_id(ablation_root, run_id)
    tag_dir = _rel_tag_dir_from_combo_dir(ablation_root, combo_dir)
    tag = str((row or {}).get("tag") or combo_dir.name)
    models_desc = (row or {}).get("models", "")
    _log(f"Located combination run_id={run_id} | tag={tag} | tag_dir={tag_dir.as_posix()}")
    if models_desc:
        _log(f"models={models_desc}")

    # Read predictions from combo_dir/<method>.
    wanted_datasets = [safe_name(x.strip().upper()) for x in _parse_list_arg(args.datasets)]
    wanted_set = {d for d in wanted_datasets if d}

    for method in methods:
        pred_root = combo_dir / method
        if not pred_root.exists():
            _log(f"Skipping method={method} because its directory is missing: {pred_root.as_posix()}")
            continue

        metrics_base = ablation_root / "04_metrics" / "ablation_ensemble" / tag_dir / method
        ensure_dir(metrics_base)

        # INTERNAL_OOF
        if "INTERNAL_OOF" in wanted_set:
            _render_one_dataset(
                pred_csv=pred_root / "internal_oof_predictions.csv",
                metrics_dir=metrics_base / "INTERNAL_OOF",
                title_suffix=f"ablation_{method}_{tag}_INTERNAL_OOF",
                overwrite=bool(args.overwrite),
            )

        # EXTERNAL_MERGE
        if "EXTERNAL_MERGE" in wanted_set:
            _render_one_dataset(
                pred_csv=pred_root / "external_predictions__EXTERNAL_MERGE.csv",
                metrics_dir=metrics_base / "EXTERNAL_MERGE",
                title_suffix=f"ablation_{method}_{tag}_EXTERNAL_MERGE",
                overwrite=bool(args.overwrite),
            )

        # Render the selected external datasets.
        for dt in sorted(wanted_set - {"INTERNAL_OOF", "EXTERNAL_MERGE"}):
            _render_one_dataset(
                pred_csv=pred_root / f"external_predictions__{safe_name(dt)}.csv",
                metrics_dir=metrics_base / safe_name(dt),
                title_suffix=f"ablation_{method}_{tag}_{safe_name(dt)}",
                overwrite=bool(args.overwrite),
            )

        # all external
        if args.all_external:
            for pred_csv in sorted(pred_root.glob("external_predictions__*.csv")):
                name = pred_csv.name
                if name == "external_predictions__EXTERNAL_MERGE.csv":
                    continue
                dt = name.replace("external_predictions__", "").replace(".csv", "")
                _render_one_dataset(
                    pred_csv=pred_csv,
                    metrics_dir=metrics_base / safe_name(dt),
                    title_suffix=f"ablation_{method}_{tag}_{safe_name(dt)}",
                    overwrite=bool(args.overwrite),
                )

    _log(f"done. combo_dir={combo_dir.as_posix()} | metrics_root={(ablation_root / '04_metrics' / 'ablation_ensemble' / tag_dir).as_posix()}")


if __name__ == "__main__":
    main()


