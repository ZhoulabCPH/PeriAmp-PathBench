"""
Extract ABMIL attention for multiple configurations, feature directories, and folds.
Write per-fold NPZ files and optionally FoldAvg NPZ files. Use --case-csv to select and order cases, or omit it to process all HDF5 bags. Plot the outputs with plot_attention_from_npz.py.
"""

import argparse
import csv
import gc
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from integrated_gradients import load_h5, load_config_and_model  # noqa: E402
from heatmap_common import (  # noqa: E402
    CATEGORY_ATTN_NPZ,
    category_path,
    collect_h5_files,
    extract_mil_name,
    format_seconds,
    match_feature_for_config,
    parse_config_csv,
    parse_feature_csv,
    sanitize,
)
from calculate_attention import (  # noqa: E402
    ATTN_NORM_MODES,
    compute_abmil_attention,
    normalize_attention,
)

# ============================================================
# Default settings
# ============================================================

# Optionally calculate FoldAvg attention.
DEFAULT_AVERAGE_FOLDS = True

# Default cross-validation folds
DEFAULT_FOLDS = [0, 1, 2, 3, 4]

# Display modes: none, minmax, rank, or log
DEFAULT_ATTN_NORM = "minmax"

# Feature directory CSV, which takes precedence over DEFAULT_FEATURE_DIR.
DEFAULT_FEATURE_CSV = r"data/run_configs/feature_all_model.csv"

# Single feature directory used when DEFAULT_FEATURE_CSV is absent.
DEFAULT_FEATURE_DIR = None

# ABMIL configuration CSV with one config.json per row
DEFAULT_CONFIG_CSV = r"data/run_configs/config_ABMIL - all.csv"

# Optional case list. None uses every HDF5 file in the feature directory.
DEFAULT_CASE_CSV: Optional[str] = r"data/run_configs/cases_annotated.csv"

# Fallback class names when the model configuration omits them.
DEFAULT_CLASS_NAMES = [
    "Cholangiocarcinoma",
    "Ampullary",
    "Duodenal",
    "Pancreatic",
]

# Root directory for attention NPZ files and summary.csv.
DEFAULT_OUTDIR = r"outputs/interpretability/attention/all_attention"

# ============================================================


def load_case_csv(csv_path: Path) -> List[str]:
    """Read unique case IDs in CSV order. Accept a headerless list or a column named case, case_id, caseid, id, patient, or patient_id."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Case CSV not found: {csv_path}")

    case_ids: List[str] = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        first_data_checked = False

        for row in reader:
            if not row:
                continue

            value = str(row[0]).strip()

            if not value:
                continue

            # Allow comment lines beginning with #.
            if value.startswith("#"):
                continue

            # Skip the first cell when it contains a supported header.
            if not first_data_checked:
                first_data_checked = True

                if value.lower() in {
                    "case",
                    "case_id",
                    "caseid",
                    "id",
                    "patient",
                    "patient_id",
                }:
                    continue

            case_ids.append(value)

    # Remove duplicate IDs while preserving CSV order.
    case_ids = list(dict.fromkeys(case_ids))

    if not case_ids:
        raise RuntimeError(f"Case CSV contains no case IDs: {csv_path}")

    return case_ids


def filter_h5_files_by_cases(
    h5_files: List[Path],
    selected_cases: Optional[List[str]],
    feature_label: str = "",
    print_missing: bool = True,
) -> List[Path]:
    """When a case list is supplied, match its IDs to HDF5 stems and retain the requested order."""
    h5_files = list(h5_files)

    if selected_cases is None:
        return h5_files

    h5_map = {p.stem: p for p in h5_files}

    selected_h5 = [
        h5_map[case_id]
        for case_id in selected_cases
        if case_id in h5_map
    ]

    missing_cases = [
        case_id
        for case_id in selected_cases
        if case_id not in h5_map
    ]

    label_text = feature_label or "(single)"

    print(
        f"[case filter] feature={label_text}: "
        f"requested {len(selected_cases)} cases, "
        f"found {len(selected_h5)} cases"
    )

    if print_missing and missing_cases:
        print(
            f"[warning] feature={label_text} has "
            f"{len(missing_cases)} cases without matching HDF5 files: "
        )
        for case_id in missing_cases:
            print(f"  - {case_id}.h5")

    return selected_h5


def run_single(
    model,
    h5_path: Path,
    outroot: Path,
    file_prefix: str,
    class_names: List[str],
    attn_norm: str,
    device: torch.device,
    model_label: Optional[str] = None,
) -> dict:
    """Extract ABMIL attention for one job and save it under attentions."""
    features_np, coords_np = load_h5(str(h5_path))
    n_patches, feat_dim = features_np.shape
    print(f"  [data] patches={n_patches}, dim={feat_dim}")

    features = torch.tensor(
        features_np,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    mask = torch.ones(
        1,
        n_patches,
        dtype=torch.bool,
    ).to(device)

    model_label = model_label or file_prefix

    with torch.no_grad():
        logits = model(features, mask)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_class = int(np.argmax(probs))

    attn = compute_abmil_attention(model, features, mask)
    attn_vis = normalize_attention(attn, attn_norm)

    print(
        f"  [attention] min={attn.min():.3e}, max={attn.max():.3e}, "
        f"sum={attn.sum():.4f}, pred={class_names[pred_class]}"
    )

    np.savez_compressed(
        category_path(
            outroot,
            CATEGORY_ATTN_NPZ,
            file_prefix,
            "npz",
        ),
        patch_attributions=attn,
        attention=attn,
        attention_vis=attn_vis,
        attn_norm=attn_norm,
        coords=coords_np,
        case_id=h5_path.stem,
        probs=probs,
        pred_class=pred_class,
        class_names=class_names,
        model_label=model_label,
        n_models=1,
    )

    print('  [output] Saved attention NPZ only.')

    attn_sorted = np.sort(attn)[::-1]
    top10_mass = float(
        attn_sorted[:max(1, n_patches // 10)].sum()
    )

    try:
        del features, mask, logits
    except NameError:
        pass

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pred_class": pred_class,
        "pred_name": class_names[pred_class],
        "probs": ";".join(f"{p:.4f}" for p in probs.tolist()),
        "attn_max": float(attn.max()),
        "attn_top10pct_mass": top10_mass,
        "n_patches": int(n_patches),
        "model_label": model_label,
        "n_models": 1,
    }


def run_average_folds_from_saved(
    h5_path: Path,
    outroot: Path,
    file_prefix_avg: str,
    file_prefix_by_fold,
    folds: List[int],
    class_names: List[str],
    attn_norm: str,
) -> dict:
    """Read per-fold attention NPZ files and write a FoldAvg NPZ."""
    attns = []
    probs_list = []
    coords_ref = None
    used_folds = []

    for fold in folds:
        prefix = file_prefix_by_fold(fold)
        npz_path = category_path(
            outroot,
            CATEGORY_ATTN_NPZ,
            prefix,
            "npz",
        )

        if not npz_path.exists():
            print(
                f"  [FoldAvg] Skipping missing attention file: "
                f"{npz_path.name}"
            )
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            attn = np.asarray(
                data["attention"]
                if "attention" in data.files
                else data["patch_attributions"],
                dtype=np.float64,
            ).reshape(-1)

            coords = np.asarray(data["coords"])
            probs = np.asarray(
                data["probs"],
                dtype=np.float64,
            ).reshape(-1)

        if coords_ref is None:
            coords_ref = coords

        elif (
            coords_ref.shape != coords.shape
            or not np.array_equal(coords_ref, coords)
        ):
            raise ValueError(
                f"FoldAvg coordinates differ for fold={fold}, "
                f"case={h5_path.stem}，"
                'Cannot average attention patch by patch.'
            )

        attns.append(attn)
        probs_list.append(probs)
        used_folds.append(int(fold))

    if not attns:
        raise FileNotFoundError(
            f"FoldAvg has no usable fold attention files: "
            f"{h5_path.stem}"
        )

    attn_avg = np.mean(
        np.stack(attns, axis=0),
        axis=0,
    )

    probs_avg = np.mean(
        np.stack(probs_list, axis=0),
        axis=0,
    )

    pred_class = int(np.argmax(probs_avg))
    attn_vis = normalize_attention(attn_avg, attn_norm)
    coords_np = coords_ref
    n_patches = int(attn_avg.shape[0])
    model_label = f"FoldAvg_n{len(used_folds)}"

    print(
        f"  [FoldAvg] used_folds={used_folds}, "
        f"min={attn_avg.min():.3e}, "
        f"max={attn_avg.max():.3e}, "
        f"sum={attn_avg.sum():.4f}, "
        f"pred={class_names[pred_class]}"
    )

    np.savez_compressed(
        category_path(
            outroot,
            CATEGORY_ATTN_NPZ,
            file_prefix_avg,
            "npz",
        ),
        patch_attributions=attn_avg,
        attention=attn_avg,
        attention_vis=attn_vis,
        attn_norm=attn_norm,
        coords=coords_np,
        case_id=h5_path.stem,
        probs=probs_avg,
        pred_class=pred_class,
        class_names=class_names,
        averaged_folds=np.asarray(
            used_folds,
            dtype=np.int64,
        ),
        model_label=model_label,
        n_models=len(used_folds),
    )

    print('  [output] Saved FoldAvg attention NPZ.')

    attn_sorted = np.sort(attn_avg)[::-1]
    top10_mass = float(
        attn_sorted[:max(1, n_patches // 10)].sum()
    )

    return {
        "pred_class": pred_class,
        "pred_name": class_names[pred_class],
        "probs": ";".join(
            f"{p:.4f}" for p in probs_avg.tolist()
        ),
        "attn_max": float(attn_avg.max()),
        "attn_top10pct_mass": top10_mass,
        "n_patches": n_patches,
        "model_label": model_label,
        "n_models": int(len(used_folds)),
        "n_folds_averaged": int(len(used_folds)),
        "averaged_folds": ";".join(
            str(x) for x in used_folds
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch ABMIL Attention: "
            'Compute attention for multiple configurations, encoder features, and folds. '
            'Save attention arrays under attentions/.'
        )
    )

    g_data = parser.add_argument_group('Data configuration')

    g_data.add_argument(
        "--feature-csv",
        type=str,
        default=DEFAULT_FEATURE_CSV,
        help=(
            'CSV with feature_path and label columns. '
            'Each row identifies one feature directory.'
        ),
    )

    g_data.add_argument(
        "--feature-dir",
        type=str,
        default=DEFAULT_FEATURE_DIR,
        help=(
            'Single directory of HDF5 feature bags. '
            'Used only when --feature-csv is omitted.'
        ),
    )

    g_data.add_argument(
        "--config-csv",
        type=str,
        default=DEFAULT_CONFIG_CSV,
        help=(
            'CSV with one config.json path per row '
            'and an optional label column.'
        ),
    )

    g_data.add_argument(
        "--case-csv",
        type=str,
        default=DEFAULT_CASE_CSV,
        help=(
            'Optional CSV with one case ID per row. '
            'Accepts a headerless list or a supported header: '
            'case, case_id, caseid, id, patient, or patient_id. '
            'Omit to process all HDF5 files.'
        ),
    )

    g_data.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=DEFAULT_FOLDS,
        help=(
            'Fold indices to process. '
            'Defaults to 0 1 2 3 4.'
        ),
    )

    g_data.add_argument(
        "--average-folds",
        dest="average_folds",
        action="store_true",
        default=DEFAULT_AVERAGE_FOLDS,
        help=(
            'Average attention over the selected folds '
            'and save FoldAvg output.'
        ),
    )

    g_data.add_argument(
        "--no-average-folds",
        dest="average_folds",
        action="store_false",
        help='Disable FoldAvg attention output',
    )

    g_data.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help='Class names to use when the configuration omits them',
    )

    g_calc = parser.add_argument_group('Output options')

    g_calc.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help='Output root for the attention NPZ files',
    )

    g_calc.add_argument(
        "--attn-norm",
        choices=list(ATTN_NORM_MODES),
        default=DEFAULT_ATTN_NORM,
        help=(
            'Normalization for display attention. '
            'The raw attention is also saved.'
        ),
    )

    args = parser.parse_args()

    # ────────────────────────────────────────────────────────
    # Optional case list
    # ────────────────────────────────────────────────────────
    selected_cases: Optional[List[str]] = None

    if args.case_csv:
        selected_cases = load_case_csv(
            Path(args.case_csv)
        )
        print(f"[Case CSV] {args.case_csv}")
        print(
            f"[case CSV] Read "
            f"{len(selected_cases)} case IDs."
        )
    else:
        print(
            '[case CSV] No list supplied. '
            'Processing every HDF5 bag in the feature directory.'
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"[device] {device}")
    print(
        '[output] Saving attentions/*.npz. '
        'Use plot_attention_from_npz.py for rendering.'
    )

    # ────────────────────────────────────────────────────────
    # Resolve feature inputs.
    # Prefer --feature-csv over --feature-dir.
    # ────────────────────────────────────────────────────────
    feature_entries: Optional[List[dict]] = None
    single_h5_files: Optional[List[Path]] = None

    if args.feature_csv:
        feature_entries = parse_feature_csv(
            Path(args.feature_csv)
        )

        if not feature_entries:
            raise ValueError(
                f"Feature CSV contains no usable entries: "
                f"{args.feature_csv}"
            )

        if args.feature_dir:
            print(
                '[config] Both --feature-csv '
                'and --feature-dir are set. '
                'Using --feature-csv.'
            )

    elif args.feature_dir:
        feature_dir = Path(args.feature_dir)

        if not feature_dir.is_dir():
            raise NotADirectoryError(
                f"--feature-dir is not a directory: "
                f"{feature_dir}"
            )

        single_h5_files = collect_h5_files(
            feature_dir
        )

        if not single_h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in "
                f"{feature_dir}"
            )

    else:
        raise ValueError(
            'Provide --feature-csv '
            'or --feature-dir.'
        )

    config_entries = parse_config_csv(
        Path(args.config_csv)
    )

    if not config_entries:
        raise ValueError(
            f"Configuration CSV contains no usable entries: "
            f"{args.config_csv}"
        )

    folds = args.folds
    outroot = Path(args.outdir)
    outroot.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ────────────────────────────────────────────────────────
    # Build the execution plan.
    # Match each configuration to its feature entry.
    # Apply the optional case list before counting jobs.
    # ────────────────────────────────────────────────────────
    plan: List[dict] = []

    for cfg_entry in config_entries:
        cfg_label = cfg_entry["label"]

        if feature_entries is not None:
            fe_original = match_feature_for_config(
                feature_entries,
                cfg_label,
            )
        else:
            fe_original = {
                "feature_path": str(args.feature_dir),
                "label": "",
                "h5_files": single_h5_files or [],
            }

        if fe_original is None:
            fe = None

        else:
            # Copy the feature entry before filtering.
            fe = dict(fe_original)

            original_h5_files = list(
                fe_original.get("h5_files", [])
            )

            filtered_h5_files = (
                filter_h5_files_by_cases(
                    h5_files=original_h5_files,
                    selected_cases=selected_cases,
                    feature_label=fe_original.get(
                        "label",
                        "",
                    ),
                    print_missing=True,
                )
            )

            fe["h5_files"] = filtered_h5_files
            fe["n_h5_before_case_filter"] = len(
                original_h5_files
            )
            fe["n_h5_after_case_filter"] = len(
                filtered_h5_files
            )

        plan.append({
            "config": cfg_entry,
            "feature": fe,
        })

    print(
        f"[config] Found {len(config_entries)} model configurations "
        f"@ {args.config_csv}"
    )

    if feature_entries is not None:
        print(
            f"[features] Found {len(feature_entries)} "
            f" encoder feature sets "
            f"@ {args.feature_csv}"
        )

        # Report the filtered counts from the plan.
        shown_features = set()

        for item in plan:
            fe = item["feature"]

            if fe is None:
                continue

            key = (
                str(fe.get("label", "")),
                str(fe.get("feature_path", "")),
            )

            if key in shown_features:
                continue

            shown_features.add(key)

            n_before = int(
                fe.get(
                    "n_h5_before_case_filter",
                    len(fe["h5_files"]),
                )
            )
            n_after = int(
                fe.get(
                    "n_h5_after_case_filter",
                    len(fe["h5_files"]),
                )
            )

            print(
                f"  · label={fe['label']:<20} "
                f"h5_total={n_before:>4} "
                f"h5_selected={n_after:>4} "
                f"dir={fe['feature_path']}"
            )

    else:
        n_before = len(
            single_h5_files or []
        )

        selected_single = (
            plan[0]["feature"]["h5_files"]
            if plan
            and plan[0]["feature"] is not None
            else []
        )

        print(
            f"[features] Single-directory mode: "
            f"{args.feature_dir} "
            f"(h5_total={n_before}, "
            f"h5_selected={len(selected_single)})"
        )

    print('[matching] Configuration to features:')

    for item in plan:
        cfg_label = item["config"]["label"]
        fe = item["feature"]

        if fe is None:
            print(
                f"  · config={cfg_label:<30} "
                f" has no matching features. Skipping."
            )

        else:
            print(
                f"  · config={cfg_label:<30} "
                f"← feature={fe['label'] or '(single)'} "
                f"(h5_selected={len(fe['h5_files'])})"
            )

    print(f"[Fold] {folds}")
    print(
        f"[FoldAvg] "
        f"{'enabled' if args.average_folds else 'disabled'}"
    )
    print('[tumor] Filtering is disabled during computation.')

    if selected_cases is None:
        print('[cases] All HDF5 bags')
    else:
        print(
            f"[cases] Requested "
            f"{len(selected_cases)} case IDs from CSV."
        )

    print(f"[output] {outroot}")

    # ────────────────────────────────────────────────────────
    # Count planned tasks.
    # Use the filtered HDF5 list.
    # ────────────────────────────────────────────────────────
    total = 0

    for item in plan:
        fe = item["feature"]

        if fe is None:
            continue

        total += (
            len(folds)
            * len(fe["h5_files"])
        )

        if (
            args.average_folds
            and len(folds) > 1
        ):
            total += len(fe["h5_files"])

    if total == 0:
        if selected_cases is not None:
            raise RuntimeError(
                'No runnable jobs. '
                'None of the requested cases may be present '
                'in the matching feature directories, '
                'or no configuration matched a feature set.'
            )

        raise RuntimeError(
            'No runnable jobs. '
            'Check configuration labels '
            'and HDF5 feature directories.'
        )

    summary: List[dict] = []
    done = 0
    start_time = time.time()

    # ────────────────────────────────────────────────────────
    # Run the planned tasks.
    # ────────────────────────────────────────────────────────
    for item in plan:
        cfg_entry = item["config"]
        fe = item["feature"]

        label = cfg_entry["label"]
        cfg_path = cfg_entry["config_path"]

        if fe is None:
            print(
                f"\n[skip] config label={label} "
                'No matching label in the feature CSV.'
            )

            summary.append({
                "case": "",
                "label": label,
                "fold": "",
                "feature_label": "",
                "error": "no_matching_feature",
            })
            continue

        feat_label = fe["label"]
        h5_files = fe["h5_files"]

        if not h5_files:
            print(
                f"\n[skip] feature label={feat_label} "
                'No HDF5 files remain after case filtering.'
            )

            summary.append({
                "case": "",
                "label": label,
                "fold": "",
                "feature_label": feat_label,
                "error": "empty_feature_dir_after_case_filter",
            })
            continue

        mil_name = extract_mil_name(
            label,
            feat_label,
        )
        mil_name_safe = sanitize(mil_name)

        feat_label_safe = (
            sanitize(feat_label)
            if feat_label
            else ""
        )

        class_names_for_average = (
            args.class_names
        )

        # ────────────────────────────────────────────────────
        # Process each fold.
        # ────────────────────────────────────────────────────
        for fold in folds:
            print(f"\n{'#' * 72}")

            print(
                f"[model] config_label={label} "
                f"| feature_label={feat_label or '(single)'} "
                f"| fold={fold}"
            )

            print(
                f"[config.json] {cfg_path}"
            )

            print(
                f"[feature_dir] {fe['feature_path']}  "
                f"(h5_selected={len(h5_files)})"
            )

            print(f"{'#' * 72}")

            try:
                model, cfg = load_config_and_model(
                    cfg_path,
                    fold,
                    device,
                )

            except Exception as e:
                print(
                    f"\n[error] Could not load model: "
                    f"label={label}, "
                    f"fold={fold}, "
                    f"config={cfg_path} → {e}"
                )

                traceback.print_exc()

                for h5_path in h5_files:
                    done += 1

                    summary.append({
                        "case": h5_path.stem,
                        "label": label,
                        "fold": fold,
                        "feature_label": feat_label,
                        "error": f"load_model: {e}",
                    })

                continue

            class_names = cfg.get(
                "class_names",
                args.class_names,
            )

            class_names_for_average = class_names

            for h5_path in h5_files:
                done += 1

                stem = h5_path.stem
                stem_safe = sanitize(stem)

                elapsed = (
                    time.time() - start_time
                )

                rate = (
                    done / elapsed
                    if elapsed > 1e-9
                    else 0.0
                )

                eta = (
                    (total - done) / rate
                    if rate > 1e-9
                    else 0.0
                )

                pct = (
                    100.0
                    * done
                    / max(total, 1)
                )

                tag = (
                    f"[progress] {done}/{total} "
                    f"({pct:.1f}%) | "
                    f"elapsed={format_seconds(elapsed)} | "
                    f"eta={format_seconds(eta)} | "
                    f"rate={rate:.2f} it/s"
                )

                print(
                    f"\n{'=' * 72}\n"
                    f"{tag}\n"
                    f"{'=' * 72}"
                )

                print(
                    f"[case] {stem} "
                    f"| config={label} "
                    f"| feature={feat_label or '(single)'} "
                    f"| fold={fold}"
                )

                if feat_label_safe:
                    file_prefix = (
                        f"{stem_safe}-"
                        f"{feat_label_safe}-"
                        f"{mil_name_safe}-"
                        f"Fold{fold}"
                    )
                else:
                    file_prefix = (
                        f"{stem_safe}-"
                        f"{mil_name_safe}-"
                        f"Fold{fold}"
                    )

                try:
                    info = run_single(
                        model=model,
                        h5_path=h5_path,
                        outroot=outroot,
                        file_prefix=file_prefix,
                        class_names=class_names,
                        attn_norm=args.attn_norm,
                        device=device,
                        model_label=f"Fold{fold}",
                    )

                    summary.append({
                        "case": stem,
                        "label": label,
                        "fold": fold,
                        **info,
                        "feature_label": feat_label,
                        "mil": mil_name,
                        "file_prefix": file_prefix,
                    })

                except Exception as e:
                    print(
                        f"  [error] Job failed: {e}"
                    )

                    traceback.print_exc()

                    summary.append({
                        "case": stem,
                        "label": label,
                        "fold": fold,
                        "feature_label": feat_label,
                        "mil": mil_name,
                        "error": str(e),
                        "file_prefix": file_prefix,
                    })

                finally:
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                        mem_alloc = (
                            torch.cuda.memory_allocated()
                            / 1024**2
                        )

                        mem_resv = (
                            torch.cuda.memory_reserved()
                            / 1024**2
                        )

                        print(
                            f"  [GPU memory] "
                            f"allocated={mem_alloc:.1f} MB, "
                            f"reserved={mem_resv:.1f} MB"
                        )

            model.zero_grad(
                set_to_none=True
            )

            del model
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ────────────────────────────────────────────────────
        # FoldAvg
        # ────────────────────────────────────────────────────
        if (
            args.average_folds
            and len(folds) > 1
        ):
            print(f"\n{'#' * 72}")

            print(
                f"[FoldAvg] config_label={label} "
                f"| feature_label={feat_label or '(single)'}"
            )

            print(
                f"[FoldAvg] folds={folds}"
            )

            print(f"{'#' * 72}")

            for h5_path in h5_files:
                done += 1

                stem = h5_path.stem
                stem_safe = sanitize(stem)

                elapsed = (
                    time.time() - start_time
                )

                rate = (
                    done / elapsed
                    if elapsed > 1e-9
                    else 0.0
                )

                eta = (
                    (total - done) / rate
                    if rate > 1e-9
                    else 0.0
                )

                pct = (
                    100.0
                    * done
                    / max(total, 1)
                )

                tag = (
                    f"[progress] {done}/{total} "
                    f"({pct:.1f}%) | "
                    f"elapsed={format_seconds(elapsed)} | "
                    f"eta={format_seconds(eta)} | "
                    f"rate={rate:.2f} it/s"
                )

                print(
                    f"\n{'=' * 72}\n"
                    f"{tag}\n"
                    f"{'=' * 72}"
                )

                print(
                    f"[case FoldAvg] {stem} "
                    f"| config={label} "
                    f"| feature={feat_label or '(single)'}"
                )

                def _prefix_for_fold(
                    fold_value
                ):
                    if feat_label_safe:
                        return (
                            f"{stem_safe}-"
                            f"{feat_label_safe}-"
                            f"{mil_name_safe}-"
                            f"Fold{fold_value}"
                        )

                    return (
                        f"{stem_safe}-"
                        f"{mil_name_safe}-"
                        f"Fold{fold_value}"
                    )

                file_prefix_avg = (
                    _prefix_for_fold("Avg")
                )

                try:
                    info = (
                        run_average_folds_from_saved(
                            h5_path=h5_path,
                            outroot=outroot,
                            file_prefix_avg=file_prefix_avg,
                            file_prefix_by_fold=_prefix_for_fold,
                            folds=folds,
                            class_names=class_names_for_average,
                            attn_norm=args.attn_norm,
                        )
                    )

                    summary.append({
                        "case": stem,
                        "label": label,
                        "fold": "Avg",
                        **info,
                        "feature_label": feat_label,
                        "mil": mil_name,
                        "file_prefix": file_prefix_avg,
                    })

                except Exception as e:
                    print(
                        f"  [error] FoldAvg "
                        f"failed: {e}"
                    )

                    traceback.print_exc()

                    summary.append({
                        "case": stem,
                        "label": label,
                        "fold": "Avg",
                        "feature_label": feat_label,
                        "mil": mil_name,
                        "error": str(e),
                        "file_prefix": file_prefix_avg,
                    })

                finally:
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────
    # Write the run summary.
    # ────────────────────────────────────────────────────────
    ok = sum(
        1
        for r in summary
        if "error" not in r
    )

    fail = len(summary) - ok

    total_elapsed = (
        time.time() - start_time
    )

    print(f"\n{'=' * 72}")

    summary_csv = (
        outroot / "summary.csv"
    )

    fields = [
        "case",
        "label",
        "feature_label",
        "mil",
        "fold",
        "pred_class",
        "pred_name",
        "n_patches",
        "attn_max",
        "attn_top10pct_mass",
        "model_label",
        "n_models",
        "n_folds_averaged",
        "averaged_folds",
        "probs",
        "file_prefix",
        "error",
    ]

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in summary:
            writer.writerow(row)

    print(f"[summary] {summary_csv}")

    print(
        f"[complete] Total jobs: {len(summary)}, "
        f"succeeded: {ok}，"
        f"failed: {fail}"
    )

    print(
        f"[elapsed] "
        f"{format_seconds(total_elapsed)}"
    )

    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
