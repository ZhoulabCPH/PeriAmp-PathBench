"""
Compute Integrated Gradients for multiple encoder directories, MIL configurations, and folds.
Use --feature-csv or --feature-dir, --config-csv, and optionally --case-csv. Write per-fold attribution NPZ files and summary.csv. An optional FoldAvg combines only folds with matching coordinates, patch counts, and target classes. Plot saved results with plot_ig_from_npz.py.
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

from integrated_gradients import (  # noqa: E402
    load_h5,
    load_config_and_model,
    integrated_gradients,
)
from heatmap_common import (  # noqa: E402
    CATEGORY_ATTR_NPZ,
    category_path,
    collect_h5_files,
    extract_mil_name,
    format_seconds,
    match_feature_for_config,
    parse_config_csv,
    parse_feature_csv,
    sanitize,
)


# ============================================================
# Default settings
# Command-line arguments override these defaults.
# ============================================================

DEFAULT_FEATURE_CSV = r"data/run_configs/feature_gigapath.csv"
DEFAULT_FEATURE_DIR = None
DEFAULT_CONFIG_CSV = r"data/run_configs/config_gigapath_abmil.csv"

# An optional case list limits which HDF5 bags are processed.
DEFAULT_CASE_CSV: Optional[str] = r"data/run_configs/cases_annotated.csv"

DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_AVERAGE_FOLDS = True

DEFAULT_CLASS_NAMES = [
    "Cholangiocarcinoma",
    "Ampullary",
    "Duodenal",
    "Pancreatic",
]

DEFAULT_TARGET_CLASS = None
DEFAULT_N_STEPS = 100
DEFAULT_IG_BATCH_SIZE = 1
DEFAULT_BASELINE = "zero"  # {zero, mean, random}

DEFAULT_OUTDIR = (
    r"outputs/interpretability/integrated_gradients"
)

# ============================================================


def load_case_csv(csv_path: Path) -> List[str]:
    """Read unique case IDs in input order, accepting a headerless list or a recognized case ID column."""
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

            if value.startswith("#"):
                continue

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
    """Match the requested IDs to HDF5 stems in CSV order. Use all available files when no case list is supplied."""
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


def build_baseline(
    features: torch.Tensor,
    baseline_type: str,
) -> torch.Tensor:
    """Construct the Integrated Gradients baseline tensor."""
    if baseline_type == "zero":
        return torch.zeros_like(features)

    if baseline_type == "mean":
        return features.mean(dim=1, keepdim=True).expand_as(features)

    if baseline_type == "random":
        # Scale the random baseline by the whole-bag standard deviation.
        std = features.std()
        return torch.randn_like(features) * std

    raise ValueError(f"Unknown baseline type: {baseline_type}")


def run_single(
    model,
    h5_path: Path,
    outroot: Path,
    file_prefix: str,
    class_names: List[str],
    target_class: Optional[int],
    n_steps: int,
    ig_batch_size: int,
    baseline_type: str,
    device: torch.device,
    model_label: Optional[str] = None,
) -> dict:
    """Compute and save attributions for one case, encoder, MIL method, and fold."""
    features_np, coords_np = load_h5(str(h5_path))

    if features_np.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional feature array (n_patches, feature_dim), "
            f"found shape={features_np.shape}"
        )

    n_patches, feat_dim = features_np.shape

    if n_patches <= 0:
        raise ValueError(f"No patches found in HDF5 file: {h5_path}")

    print(f"  [data] patches={n_patches}, dim={feat_dim}")

    features = torch.as_tensor(
        features_np,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    mask = torch.ones(
        1,
        n_patches,
        dtype=torch.bool,
        device=device,
    )

    model_label = model_label or file_prefix

    with torch.no_grad():
        logits = model(features, mask)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                f"Expected logits with shape (1, n_classes), "
                f"found shape={tuple(logits.shape)}"
            )

        probs = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze(0)
        pred_class = int(np.argmax(probs))

    n_classes = int(probs.shape[0])

    if len(class_names) != n_classes:
        raise ValueError(
            f"Number of class names ({len(class_names)}) differs from the model class count ("
            f"({n_classes})."
        )

    tgt = pred_class if target_class is None else int(target_class)

    if not (0 <= tgt < n_classes):
        raise ValueError(
            f"target_class={tgt} is outside the available classes. Class count={n_classes}"
        )

    baseline = build_baseline(features, baseline_type)

    print(
        f"  [IG] steps={n_steps}, baseline={baseline_type}, "
        f"target={class_names[tgt]} ({tgt})"
    )

    patch_attr, delta = integrated_gradients(
        model=model,
        features=features,
        mask=mask,
        target_class=tgt,
        baseline=baseline,
        n_steps=n_steps,
        batch_size=ig_batch_size,
        device=device,
    )

    patch_attr = np.asarray(patch_attr).reshape(-1)

    if patch_attr.shape[0] != n_patches:
        raise ValueError(
            f"Patch attribution count ({patch_attr.shape[0]}) "
            f" differs from patch count ({n_patches})."
        )

    delta = float(delta)

    print(
        f"  [IG complete] delta={delta:.4e}, "
        f"mean={patch_attr.mean():.4e}, "
        f"std={patch_attr.std():.4e}, "
        f"|max|={np.abs(patch_attr).max():.4e}"
    )

    out_path = category_path(
        outroot,
        CATEGORY_ATTR_NPZ,
        file_prefix,
        "npz",
    )

    np.savez_compressed(
        out_path,
        patch_attributions=patch_attr,
        coords=coords_np,
        case_id=h5_path.stem,
        probs=probs,
        pred_class=pred_class,
        target_class=tgt,
        class_names=np.asarray(class_names),
        convergence_delta=delta,
        baseline_type=baseline_type,
        n_steps=int(n_steps),
        ig_batch_size=int(ig_batch_size),
        model_label=model_label,
        n_models=1,
    )

    print(f"  [output] {out_path}")

    abs_attr = np.abs(patch_attr)
    top_k = max(1, n_patches // 10)
    top10_abs_mass = float(
        np.sort(abs_attr)[::-1][:top_k].sum()
        / max(abs_attr.sum(), 1e-12)
    )

    try:
        del baseline, features, mask, logits
    except NameError:
        pass

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "pred_class": pred_class,
        "pred_name": class_names[pred_class],
        "target_class": tgt,
        "target_name": class_names[tgt],
        "probs": ";".join(f"{p:.6f}" for p in probs.tolist()),
        "convergence_delta": delta,
        "attr_mean": float(patch_attr.mean()),
        "attr_std": float(patch_attr.std()),
        "attr_abs_max": float(abs_attr.max()),
        "attr_top10pct_abs_mass": top10_abs_mass,
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
) -> dict:
    """Average saved fold attributions per tile when coordinates, lengths, and target classes agree."""
    attrs = []
    probs_list = []
    deltas = []
    coords_ref = None
    target_ref = None
    used_folds = []
    pred_classes = []
    baseline_types = []
    n_steps_values = []

    for fold in folds:
        prefix = file_prefix_by_fold(fold)
        npz_path = category_path(
            outroot,
            CATEGORY_ATTR_NPZ,
            prefix,
            "npz",
        )

        if not npz_path.exists():
            print(f"  [FoldAvg] Skipping missing attribution file: {npz_path.name}")
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            attr = np.asarray(
                data["patch_attributions"],
                dtype=np.float64,
            ).reshape(-1)

            coords = np.asarray(data["coords"])
            probs = np.asarray(
                data["probs"],
                dtype=np.float64,
            ).reshape(-1)

            target = int(np.asarray(data["target_class"]).item())
            pred = int(np.asarray(data["pred_class"]).item())

            delta = (
                float(np.asarray(data["convergence_delta"]).item())
                if "convergence_delta" in data.files
                else np.nan
            )

            baseline_type = (
                str(np.asarray(data["baseline_type"]).item())
                if "baseline_type" in data.files
                else ""
            )

            n_steps = (
                int(np.asarray(data["n_steps"]).item())
                if "n_steps" in data.files
                else -1
            )

        if coords_ref is None:
            coords_ref = coords
        elif (
            coords_ref.shape != coords.shape
            or not np.array_equal(coords_ref, coords)
        ):
            raise ValueError(
                f"FoldAvg coordinates differ for fold={fold}, "
                f"case={h5_path.stem}. Cannot average attribution patch by patch."
            )

        if attrs and attrs[0].shape != attr.shape:
            raise ValueError(
                f"FoldAvg attribution length differs for fold={fold}, "
                f"case={h5_path.stem}"
            )

        if target_ref is None:
            target_ref = target
        elif target_ref != target:
            raise ValueError(
                f"FoldAvg target_class mismatch: "
                f"Existing target={target_ref}, fold={fold} target={target}。"
                f"Without --target-class, folds can explain different predicted classes. "
                f"FoldAvg requires a shared target. "
                f"Supply one target with --target-class."
            )

        attrs.append(attr)
        probs_list.append(probs)
        deltas.append(delta)
        pred_classes.append(pred)
        baseline_types.append(baseline_type)
        n_steps_values.append(n_steps)
        used_folds.append(int(fold))

    if not attrs:
        raise FileNotFoundError(
            f"FoldAvg has no usable fold attributions: {h5_path.stem}"
        )

    attr_avg = np.mean(
        np.stack(attrs, axis=0),
        axis=0,
    )

    probs_avg = np.mean(
        np.stack(probs_list, axis=0),
        axis=0,
    )

    pred_class = int(np.argmax(probs_avg))
    target_class = int(target_ref)
    coords_np = coords_ref
    n_patches = int(attr_avg.shape[0])
    model_label = f"FoldAvg_n{len(used_folds)}"

    finite_deltas = np.asarray(deltas, dtype=np.float64)
    delta_avg = (
        float(np.nanmean(finite_deltas))
        if np.any(np.isfinite(finite_deltas))
        else np.nan
    )

    print(
        f"  [FoldAvg] used_folds={used_folds}, "
        f"target={class_names[target_class]} ({target_class}), "
        f"mean={attr_avg.mean():.3e}, "
        f"std={attr_avg.std():.3e}, "
        f"|max|={np.abs(attr_avg).max():.3e}, "
        f"pred(avg_prob)={class_names[pred_class]}"
    )

    out_path = category_path(
        outroot,
        CATEGORY_ATTR_NPZ,
        file_prefix_avg,
        "npz",
    )

    np.savez_compressed(
        out_path,
        patch_attributions=attr_avg,
        coords=coords_np,
        case_id=h5_path.stem,
        probs=probs_avg,
        pred_class=pred_class,
        target_class=target_class,
        class_names=np.asarray(class_names),
        convergence_delta=delta_avg,
        convergence_delta_by_fold=np.asarray(
            deltas,
            dtype=np.float64,
        ),
        averaged_folds=np.asarray(
            used_folds,
            dtype=np.int64,
        ),
        pred_class_by_fold=np.asarray(
            pred_classes,
            dtype=np.int64,
        ),
        baseline_type=(
            baseline_types[0]
            if len(set(baseline_types)) == 1
            else "mixed"
        ),
        n_steps=(
            n_steps_values[0]
            if len(set(n_steps_values)) == 1
            else -1
        ),
        model_label=model_label,
        n_models=len(used_folds),
    )

    print(f"  [output] {out_path}")

    abs_attr = np.abs(attr_avg)
    top_k = max(1, n_patches // 10)
    top10_abs_mass = float(
        np.sort(abs_attr)[::-1][:top_k].sum()
        / max(abs_attr.sum(), 1e-12)
    )

    return {
        "pred_class": pred_class,
        "pred_name": class_names[pred_class],
        "target_class": target_class,
        "target_name": class_names[target_class],
        "probs": ";".join(f"{p:.6f}" for p in probs_avg.tolist()),
        "convergence_delta": delta_avg,
        "attr_mean": float(attr_avg.mean()),
        "attr_std": float(attr_avg.std()),
        "attr_abs_max": float(abs_attr.max()),
        "attr_top10pct_abs_mass": top10_abs_mass,
        "n_patches": n_patches,
        "model_label": model_label,
        "n_models": int(len(used_folds)),
        "n_folds_averaged": int(len(used_folds)),
        "averaged_folds": ";".join(str(x) for x in used_folds),
    }


def print_gpu_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return

    mem_alloc = torch.cuda.memory_allocated() / 1024**2
    mem_resv = torch.cuda.memory_reserved() / 1024**2

    print(
        f"  [GPU memory] allocated={mem_alloc:.1f} MB, "
        f"reserved={mem_resv:.1f} MB"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch Integrated Gradients: "
            'Compute Integrated Gradients for multiple configurations, feature sets, and folds. '
            'Supports a case list and FoldAvg output.'
        )
    )

    g_data = parser.add_argument_group('Data configuration')

    g_data.add_argument(
        "--feature-csv",
        type=str,
        default=DEFAULT_FEATURE_CSV,
        help=(
            'CSV with feature_path and label columns. '
            'Takes precedence over --feature-dir.'
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
        help='CSV with one config.json path per row and an optional label column',
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
        help='Fold indices to process. Defaults to 0 1 2 3 4.',
    )

    g_data.add_argument(
        "--average-folds",
        dest="average_folds",
        action="store_true",
        default=DEFAULT_AVERAGE_FOLDS,
        help=(
            'Average saved attributions across the selected folds '
            'and write a FoldAvg NPZ.'
        ),
    )

    g_data.add_argument(
        "--no-average-folds",
        dest="average_folds",
        action="store_false",
        help='Disable FoldAvg attribution output',
    )

    g_data.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help='Class names to use when the configuration omits them',
    )

    g_ig = parser.add_argument_group('Integrated Gradients options')

    g_ig.add_argument(
        "--target-class",
        type=int,
        default=DEFAULT_TARGET_CLASS,
        help=(
            'IG target class. Defaults to the predicted class for each fold. '
            'Set one target for all folds when fold predictions differ.'
        ),
    )

    g_ig.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help='Number of steps along the IG integration path',
    )

    g_ig.add_argument(
        "--ig-batch-size",
        type=int,
        default=DEFAULT_IG_BATCH_SIZE,
        help='Interpolation points per batch',
    )

    g_ig.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE,
        choices=["zero", "mean", "random"],
        help='Integrated Gradients baseline type',
    )

    g_output = parser.add_argument_group('Output options')

    g_output.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help='Output root directory',
    )

    args = parser.parse_args()

    if args.n_steps <= 0:
        raise ValueError('--n-steps must be positive.')

    if args.ig_batch_size <= 0:
        raise ValueError('--ig-batch-size must be positive.')

    if not args.folds:
        raise ValueError('Provide at least one fold.')

    # Deduplicate folds in the requested order.
    folds = list(dict.fromkeys(int(x) for x in args.folds))

    # Optional case selection
    selected_cases: Optional[List[str]] = None

    if args.case_csv:
        selected_cases = load_case_csv(Path(args.case_csv))
        print(f"[Case CSV] {args.case_csv}")
        print(f"[case CSV] Read {len(selected_cases)} case IDs.")
    else:
        print('[case CSV] No list supplied. Processing every HDF5 bag in the feature directory.')

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[device] {device}")
    print(
        '[output] Saving attribution NPZ files and summary.csv. '
        'Use plot_ig_from_npz.py for rendering.'
    )

    # Resolve feature inputs.
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
                '[config] Both --feature-csv and --feature-dir are set. '
                'Using --feature-csv.'
            )

    elif args.feature_dir:
        feature_dir = Path(args.feature_dir)

        if not feature_dir.is_dir():
            raise NotADirectoryError(
                f"--feature-dir is not a directory: {feature_dir}"
            )

        single_h5_files = collect_h5_files(feature_dir)

        if not single_h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in {feature_dir}"
            )

    else:
        raise ValueError(
            'Provide --feature-csv or --feature-dir.'
        )

    config_entries = parse_config_csv(
        Path(args.config_csv)
    )

    if not config_entries:
        raise ValueError(
            f"Configuration CSV contains no usable entries: {args.config_csv}"
        )

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # Match configurations to features, then apply case selection.
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
            fe = dict(fe_original)

            original_h5_files = list(
                fe_original.get("h5_files", [])
            )

            filtered_h5_files = filter_h5_files_by_cases(
                h5_files=original_h5_files,
                selected_cases=selected_cases,
                feature_label=fe_original.get("label", ""),
                print_missing=True,
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

    # Report the execution plan.
    print(
        f"[config] Found {len(config_entries)} model configurations "
        f"@ {args.config_csv}"
    )

    if feature_entries is not None:
        print(
            f"[features] Found {len(feature_entries)} encoder feature sets "
            f"@ {args.feature_csv}"
        )

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
        n_before = len(single_h5_files or [])
        selected_single = (
            plan[0]["feature"]["h5_files"]
            if plan and plan[0]["feature"] is not None
            else []
        )

        print(
            f"[features] Single-directory mode: {args.feature_dir} "
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
    print(
        f"[IG] target_class={args.target_class}, "
        f"steps={args.n_steps}, "
        f"batch={args.ig_batch_size}, "
        f"baseline={args.baseline}"
    )
    print('[compute] Integrated Gradients uses all patches.')

    if selected_cases is None:
        print('[cases] All HDF5 bags')
    else:
        print(
            f"[cases] Requested "
            f"{len(selected_cases)} case IDs from CSV."
        )

    print(f"[output] {outroot}")

    if (
        args.average_folds
        and len(folds) > 1
        and args.target_class is None
    ):
        print(
            '[warning] FoldAvg is enabled without an explicit target. '
            'Folds may explain different predicted classes. '
            'To average folds, set the same target with '
            "--target-class。"
        )

    # Count the planned tasks.
    total = 0

    for item in plan:
        fe = item["feature"]

        if fe is None:
            continue

        total += len(folds) * len(fe["h5_files"])

        if args.average_folds and len(folds) > 1:
            total += len(fe["h5_files"])

    if total == 0:
        if selected_cases is not None:
            raise RuntimeError(
                'No runnable jobs. Check that the requested cases exist '
                'and the configuration labels match feature entries.'
            )

        raise RuntimeError(
            'No runnable jobs. Check model and feature labels and HDF5 inputs.'
        )

    summary: List[dict] = []
    done = 0
    start_time = time.time()

    # Run the planned tasks.
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

        class_names_for_average = args.class_names

        # Process one fold.
        for fold in folds:
            print(f"\n{'#' * 72}")
            print(
                f"[model] config_label={label} "
                f"| feature_label={feat_label or '(single)'} "
                f"| fold={fold}"
            )
            print(f"[config.json] {cfg_path}")
            print(
                f"[feature_dir] {fe['feature_path']} "
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
                    f"label={label}, fold={fold}, "
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
                        "mil": mil_name,
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

                elapsed = time.time() - start_time
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
                pct = 100.0 * done / max(total, 1)

                tag = (
                    f"[progress] {done}/{total} ({pct:.1f}%) | "
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
                        target_class=args.target_class,
                        n_steps=args.n_steps,
                        ig_batch_size=args.ig_batch_size,
                        baseline_type=args.baseline,
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
                    print(f"  [error] Job failed: {e}")
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
                        print_gpu_memory(device)

            model.zero_grad(set_to_none=True)
            del model
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── FoldAvg ──
        if args.average_folds and len(folds) > 1:
            print(f"\n{'#' * 72}")
            print(
                f"[FoldAvg] config_label={label} "
                f"| feature_label={feat_label or '(single)'}"
            )
            print(f"[FoldAvg] folds={folds}")
            print(f"{'#' * 72}")

            for h5_path in h5_files:
                done += 1

                stem = h5_path.stem
                stem_safe = sanitize(stem)

                elapsed = time.time() - start_time
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
                pct = 100.0 * done / max(total, 1)

                tag = (
                    f"[progress] {done}/{total} ({pct:.1f}%) | "
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

                def _prefix_for_fold(fold_value):
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

                file_prefix_avg = _prefix_for_fold("Avg")

                try:
                    info = run_average_folds_from_saved(
                        h5_path=h5_path,
                        outroot=outroot,
                        file_prefix_avg=file_prefix_avg,
                        file_prefix_by_fold=_prefix_for_fold,
                        folds=folds,
                        class_names=class_names_for_average,
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
                        f"  [error] FoldAvg failed: {e}"
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

    # ── summary.csv ──
    ok = sum(
        1 for r in summary
        if "error" not in r
    )
    fail = len(summary) - ok

    total_elapsed = time.time() - start_time

    print(f"\n{'=' * 72}")

    summary_csv = outroot / "summary.csv"

    fields = [
        "case",
        "label",
        "feature_label",
        "mil",
        "fold",
        "pred_class",
        "pred_name",
        "target_class",
        "target_name",
        "n_patches",
        "attr_mean",
        "attr_std",
        "attr_abs_max",
        "attr_top10pct_abs_mass",
        "convergence_delta",
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
        encoding="utf-8-sig",
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
        f"succeeded: {ok}, failed: {fail}"
    )
    print(
        f"[elapsed] {format_seconds(total_elapsed)}"
    )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
