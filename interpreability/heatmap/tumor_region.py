"""
Shared filters for tumor annotations produced by the GeoJSON tiler.
Read tile_x, tile_y, and in_annotation from the per-slide CSV. Match the CSV by case stem or provide its path explicitly. Use filter to draw tumor tiles only, or zero_outside to keep tissue tiles visible with default values outside the annotated region.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np


# Match the GeoJSON tiler's default output directory.
DEFAULT_ANNOTATION_DIR = Path(
    r"outputs/interpretability/tumor_area_output"
)
DEFAULT_ANNOTATION_CSV = None
DEFAULT_TUMOR_FILTER = True
DEFAULT_TUMOR_DISPLAY_MODE = "zero_outside"
TUMOR_DISPLAY_MODES = ("filter", "zero_outside")

CSV_SUFFIX = "_patch_annotation.csv"
SLIDE_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".mdsx")
PATCH_XY_RE = re.compile(r"(\d+)x_(\d+)y", re.IGNORECASE)

PathLike = Union[str, Path]


def normalize_case_id(case_id: str) -> str:
    """Remove the slide extension to obtain the HDF5 stem."""
    name = Path(str(case_id)).name
    lower = name.lower()
    for ext in SLIDE_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    if name.lower().endswith(".h5"):
        return Path(name).stem
    return name


def _truthy_in_annotation(value) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y")


def find_annotation_csv(
    case_id: str,
    annotation_dir: Optional[PathLike] = None,
    annotation_csv: Optional[PathLike] = None,
) -> Optional[Path]:
    """Find the patch annotation CSV for the requested case."""
    if annotation_csv is not None and str(annotation_csv).strip():
        path = Path(annotation_csv)
        if path.is_file() and path.stat().st_size > 0:
            return path
        raise FileNotFoundError(f"Annotation CSV is missing or empty: {path}")

    if annotation_dir is None or not str(annotation_dir).strip():
        return None
    root = Path(annotation_dir)
    if not root.is_dir():
        return None

    stem = normalize_case_id(case_id)
    exact_names = [f"{stem}{ext}{CSV_SUFFIX}" for ext in SLIDE_EXTS]
    exact_names.append(f"{stem}{CSV_SUFFIX}")
    for name in exact_names:
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    def _pick(matches: Sequence[Path]) -> Optional[Path]:
        files = [p for p in matches if p.is_file() and p.stat().st_size > 0]
        if not files:
            return None
        if len(files) == 1:
            return files[0]
        with_ext = [
            p for p in files
            if p.name.lower().startswith(stem.lower() + ".")
        ]
        chosen = with_ext[0] if with_ext else files[0]
        print(f"[tumor] case={stem} matches several CSVs. Using {chosen.name}")
        return chosen

    picked = _pick(sorted(root.glob(f"{stem}*{CSV_SUFFIX}")))
    if picked is not None:
        return picked
    return _pick(sorted(root.rglob(f"{stem}*{CSV_SUFFIX}")))


def load_tumor_coord_set(csv_path: PathLike) -> set:
    """Read the tile coordinates where in_annotation equals 1."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Annotation CSV has no header: {csv_path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}
        if "in_annotation" not in fields:
            raise ValueError(f"Annotation CSV is missing in_annotation: {csv_path}")
        x_key = fields.get("tile_x") or fields.get("x")
        y_key = fields.get("tile_y") or fields.get("y")
        if x_key is None or y_key is None:
            raise ValueError(f"Annotation CSV is missing tile_x or tile_y: {csv_path}")

        in_key = fields["in_annotation"]
        coords = set()
        n_total = 0
        for row in reader:
            n_total += 1
            if not _truthy_in_annotation(row.get(in_key)):
                continue
            try:
                coords.add((int(float(row[x_key])), int(float(row[y_key]))))
            except (TypeError, ValueError):
                continue

    if n_total == 0:
        raise ValueError(f"Annotation CSV has no data rows: {csv_path}")
    return coords


def tumor_coord_mask(coords: np.ndarray, tumor_xy: set) -> np.ndarray:
    """Return a Boolean mask of coordinates inside the annotation."""
    arr = np.asarray(coords)
    if arr.size == 0:
        return np.zeros((0,), dtype=bool)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected coordinates with shape (N, 2+) and found {arr.shape}")
    xy = np.rint(arr[:, :2]).astype(np.int64, copy=False)
    return np.fromiter(
        ((int(x), int(y)) in tumor_xy for x, y in xy),
        dtype=bool,
        count=xy.shape[0],
    )


def _subset_first_axis(value, mask: np.ndarray):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Array length {arr.shape[0]} differs from mask length {mask.shape[0]}."
        )
    return arr[mask]


def apply_tumor_filter(
    coords: np.ndarray,
    case_id: str,
    *,
    arrays: Optional[Dict[str, object]] = None,
    annotation_csv: Optional[PathLike] = None,
    annotation_dir: Optional[PathLike] = None,
    enabled: bool = True,
    require: bool = True,
    empty_ok: bool = False,
    log: bool = True,
) -> Tuple[np.ndarray, Dict[str, object], dict]:
    """Filter coordinates and row-aligned arrays using the tumor CSV. Return the original arrays when disabled. If required, raise on missing annotations or an empty selection."""
    arrays = dict(arrays or {})
    n_before = int(np.asarray(coords).shape[0])
    info = {
        "enabled": bool(enabled),
        "csv_path": None,
        "n_before": n_before,
        "n_after": n_before,
        "n_tumor_in_csv": 0,
        "case_id": normalize_case_id(case_id),
    }
    if not enabled:
        return np.asarray(coords), arrays, info

    csv_path = find_annotation_csv(
        case_id, annotation_dir=annotation_dir, annotation_csv=annotation_csv
    )
    if csv_path is None:
        message = (
            f"Tumor annotation CSV not found for case={info['case_id']} "
            f"(annotation_dir={annotation_dir!s})"
        )
        if require:
            raise FileNotFoundError(message)
        if log:
            print(f"[warning] {message}. Processing every patch.")
        return np.asarray(coords), arrays, info

    tumor_xy = load_tumor_coord_set(csv_path)
    mask = tumor_coord_mask(coords, tumor_xy)
    n_after = int(mask.sum())
    info.update({
        "csv_path": csv_path,
        "n_after": n_after,
        "n_tumor_in_csv": len(tumor_xy),
    })
    if log:
        print(
            f"[tumor] {csv_path.name}: kept {n_after}/{n_before} patches "
            f"(annotated tiles: {len(tumor_xy)})"
        )
    if n_after == 0 and not empty_ok:
        raise ValueError(
            f"No patches matched the tumor annotation for case={info['case_id']}, "
            f"csv={csv_path}, h5/npz patches={n_before}, "
            f"Tiles marked in_annotation=1: {len(tumor_xy)}"
        )

    coords_kept = np.asarray(coords)[mask]
    arrays_kept = {key: _subset_first_axis(val, mask) for key, val in arrays.items()}
    return coords_kept, arrays_kept, info


def filter_image_paths(
    image_paths: Iterable[PathLike],
    case_id: str,
    *,
    annotation_csv: Optional[PathLike] = None,
    annotation_dir: Optional[PathLike] = None,
    enabled: bool = True,
    require: bool = True,
    log: bool = True,
) -> List[Path]:
    """Filter tile images by annotation membership using coordinates encoded in the filename."""
    paths = [Path(p) for p in image_paths]
    if not enabled:
        return paths

    csv_path = find_annotation_csv(
        case_id, annotation_dir=annotation_dir, annotation_csv=annotation_csv
    )
    if csv_path is None:
        message = (
            f"Tumor annotation CSV not found for case={normalize_case_id(case_id)} "
            f"(annotation_dir={annotation_dir!s})"
        )
        if require:
            raise FileNotFoundError(message)
        if log:
            print(f"[warning] {message}. Using every image.")
        return paths

    tumor_xy = load_tumor_coord_set(csv_path)
    kept: List[Path] = []
    unmatched_name = 0
    for path in paths:
        match = PATCH_XY_RE.search(path.name)
        if match is None:
            unmatched_name += 1
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in tumor_xy:
            kept.append(path)
    if log:
        extra = f"，{unmatched_name} images skipped because coordinates could not be parsed." if unmatched_name else ""
        print(
            f"[tumor] {csv_path.name}: kept images {len(kept)}/{len(paths)}{extra}"
        )
    return kept


def apply_tumor_filter_by_case(
    coords: np.ndarray,
    case_ids: np.ndarray,
    *,
    arrays: Optional[Dict[str, object]] = None,
    per_case_info: Optional[Sequence[dict]] = None,
    annotation_csv: Optional[PathLike] = None,
    annotation_dir: Optional[PathLike] = None,
    enabled: bool = True,
    require: bool = True,
    empty_ok: bool = False,
    log: bool = True,
) -> Tuple[np.ndarray, Dict[str, object], np.ndarray, dict]:
    """Filter the combined arrays using each case's annotation CSV. Return retained arrays, selection mask, and metadata."""
    arrays = dict(arrays or {})
    coords = np.asarray(coords)
    case_ids = np.asarray(case_ids, dtype=object).reshape(-1)
    n_before = coords.shape[0]
    mask = np.ones(n_before, dtype=bool)
    info = {
        "enabled": bool(enabled),
        "n_before": n_before,
        "n_after": n_before,
        "n_cases": 0,
        "n_cases_empty": 0,
    }
    if not enabled:
        return coords, arrays, mask, info
    if case_ids.shape[0] != n_before:
        raise ValueError('case_ids and coords have different lengths.')

    spans = list(per_case_info) if per_case_info else rebuild_case_spans(case_ids)
    info["n_cases"] = len(spans)
    for span in spans:
        start = int(span["start_row"])
        end = int(span["end_row"])
        case = str(span.get("case") or case_ids[start])
        sub = coords[start:end]
        csv_path = find_annotation_csv(
            case, annotation_dir=annotation_dir, annotation_csv=annotation_csv
        )
        if csv_path is None:
            message = (
                f"Tumor annotation CSV not found for case={normalize_case_id(case)} "
                f"(annotation_dir={annotation_dir!s})"
            )
            if require:
                raise FileNotFoundError(message)
            if log:
                print(f"[warning] {message}. Keeping every patch for this case.")
            mask[start:end] = True
            continue
        tumor_xy = load_tumor_coord_set(csv_path)
        sub_mask = tumor_coord_mask(sub, tumor_xy)
        mask[start:end] = sub_mask
        n_keep = int(sub_mask.sum())
        if log:
            print(
                f"[tumor] {case}: kept {n_keep}/{sub.shape[0]}  "
                f"({csv_path.name})"
            )
        if n_keep == 0:
            info["n_cases_empty"] += 1

    n_after = int(mask.sum())
    info["n_after"] = n_after
    if n_after == 0 and not empty_ok:
        raise ValueError('No patches passed the tumor annotation filter.')
    coords_kept = coords[mask]
    arrays_kept = {key: _subset_first_axis(val, mask) for key, val in arrays.items()}
    return coords_kept, arrays_kept, mask, info


def rebuild_case_spans(case_ids: np.ndarray) -> List[dict]:
    """Rebuild case row spans after filtering."""
    ids = np.asarray(case_ids, dtype=object).reshape(-1)
    info: List[dict] = []
    n = ids.shape[0]
    if n == 0:
        return info
    start = 0
    current = str(ids[0])
    for i in range(1, n + 1):
        if i == n or str(ids[i]) != current:
            info.append({
                "case": current,
                "n_patches": i - start,
                "start_row": start,
                "end_row": i,
            })
            if i < n:
                current = str(ids[i])
                start = i
    return info


def remap_indices_after_mask(indices: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map existing indices into the filtered arrays and discard indices removed by the mask."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return indices
    new_index = np.full(mask.shape[0], -1, dtype=np.int64)
    new_index[mask] = np.arange(int(mask.sum()), dtype=np.int64)
    mapped = new_index[indices]
    return mapped[mapped >= 0]


class TumorDisplayResult(NamedTuple):
    """Hold coordinates and values for tumor heatmaps, overlays, and dot maps."""

    plot_coords: np.ndarray
    plot_values: np.ndarray
    tumor_coords: np.ndarray
    tumor_values: np.ndarray
    tumor_indices: np.ndarray
    clip_lo: float
    clip_hi: float
    extras_plot: Dict[str, np.ndarray]
    extras_tumor: Dict[str, np.ndarray]
    filter_info: dict
    display_mode: str


def _normalize_display_mode(display_mode: Optional[str]) -> str:
    mode = str(
        DEFAULT_TUMOR_DISPLAY_MODE if display_mode is None else display_mode
    ).strip().lower()
    if mode not in TUMOR_DISPLAY_MODES:
        raise ValueError(
            f"Unknown tumor_display_mode: {display_mode!r}，"
            f"Choose from {list(TUMOR_DISPLAY_MODES)}"
        )
    return mode


def expand_first_axis(
    tumor_arr,
    n_full: int,
    tumor_indices: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Expand the tumor subset to full tissue length, using the supplied fill value elsewhere."""
    arr = np.asarray(tumor_arr)
    indices = np.asarray(tumor_indices, dtype=np.int64).reshape(-1)
    if arr.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Array length {arr.shape[0]} differs from tumor_indices length "
            f"{indices.shape[0]}."
        )
    shape = (int(n_full),) + arr.shape[1:]
    if np.issubdtype(arr.dtype, np.floating):
        out = np.full(shape, fill, dtype=arr.dtype)
    elif np.issubdtype(arr.dtype, np.integer):
        fill_i = -1 if not np.isfinite(fill) else int(fill)
        out = np.full(shape, fill_i, dtype=arr.dtype)
    elif arr.dtype == bool:
        out = np.zeros(shape, dtype=bool)
    else:
        out = np.empty(shape, dtype=object)
        out[:] = fill
    out[indices] = arr
    return out


def expand_tumor_plot_arrays(
    all_coords: np.ndarray,
    tumor_coords: np.ndarray,
    tumor_indices: np.ndarray,
    arrays: Optional[Dict[str, object]] = None,
    *,
    enabled: bool,
    display_mode: str = DEFAULT_TUMOR_DISPLAY_MODE,
    fill: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Select tumor-only or full-tissue coordinates and values using the requested display mode."""
    mode = _normalize_display_mode(display_mode)
    kept = {
        key: np.asarray(val)
        for key, val in dict(arrays or {}).items()
    }
    if (not enabled) or mode == "filter":
        return np.asarray(tumor_coords), kept

    n_full = int(np.asarray(all_coords).shape[0])
    expanded = {
        key: expand_first_axis(val, n_full, tumor_indices, fill=fill)
        for key, val in kept.items()
    }
    return np.asarray(all_coords), expanded


def percentile_of_value(values, target: float) -> float:
    """Invert NumPy's linear percentile interpolation to locate a value in the distribution."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(target):
        return float("nan")
    arr = np.sort(arr)
    n = arr.size
    if n == 1 or target <= arr[0]:
        return 0.0
    if target >= arr[-1]:
        return 100.0
    # Use a right-side search to find the end of a repeated-value interval.
    idx = int(np.searchsorted(arr, target, side="right")) - 1
    lo = float(arr[idx])
    hi = float(arr[idx + 1])
    frac = 0.0 if hi <= lo else (target - lo) / (hi - lo)
    return float(np.clip(100.0 * (idx + frac) / (n - 1), 0.0, 100.0))


def equivalent_clip_percentiles(
    plot_values,
    tumor_values,
    clip_lo: float,
    clip_hi: float,
) -> Tuple[float, float]:
    """Convert tumor-only clipping bounds to equivalent percentiles in the expanded array so outside-tumor fill values do not shift the color limits."""
    tumor = np.asarray(tumor_values, dtype=np.float64).reshape(-1)
    tumor = tumor[np.isfinite(tumor)]
    plot = np.asarray(plot_values, dtype=np.float64).reshape(-1)
    if tumor.size == 0 or plot[np.isfinite(plot)].size == 0:
        return 0.0, 100.0

    tumor_vmin = float(np.percentile(tumor, float(clip_lo)))
    tumor_vmax = float(np.percentile(tumor, float(clip_hi)))
    p_lo = percentile_of_value(plot, tumor_vmin)
    p_hi = percentile_of_value(plot, tumor_vmax)
    if not (np.isfinite(p_lo) and np.isfinite(p_hi)):
        return 0.0, 100.0
    if p_hi < p_lo:
        p_lo, p_hi = p_hi, p_lo
    return float(p_lo), float(p_hi)


def compose_tumor_display(
    all_coords: np.ndarray,
    tumor_coords: np.ndarray,
    tumor_indices: np.ndarray,
    tumor_display_values: np.ndarray,
    *,
    enabled: bool,
    display_mode: str = DEFAULT_TUMOR_DISPLAY_MODE,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
    extras_tumor: Optional[Dict[str, object]] = None,
    fill: float = 0.0,
    filter_info: Optional[dict] = None,
    log: bool = True,
) -> TumorDisplayResult:
    """Prepare already-filtered coordinates and values for visualization."""
    mode = _normalize_display_mode(display_mode)
    tumor_coords = np.asarray(tumor_coords)
    tumor_values = np.asarray(tumor_display_values, dtype=np.float64).reshape(-1)
    extras_tumor_arr = {
        key: np.asarray(val)
        for key, val in dict(extras_tumor or {}).items()
    }
    info = dict(filter_info or {})
    info.setdefault("enabled", bool(enabled))

    if not enabled:
        if log:
            print(f"[region] All tissue patches={tumor_values.size}")
        return TumorDisplayResult(
            plot_coords=tumor_coords,
            plot_values=tumor_values,
            tumor_coords=tumor_coords,
            tumor_values=tumor_values,
            tumor_indices=np.asarray(tumor_indices, dtype=np.int64).reshape(-1),
            clip_lo=float(clip_lo),
            clip_hi=float(clip_hi),
            extras_plot=extras_tumor_arr,
            extras_tumor=extras_tumor_arr,
            filter_info=info,
            display_mode=mode,
        )

    plot_arrays = {"values": tumor_values, **extras_tumor_arr}
    plot_coords, plot_arrays = expand_tumor_plot_arrays(
        all_coords,
        tumor_coords,
        tumor_indices,
        plot_arrays,
        enabled=True,
        display_mode=mode,
        fill=fill,
    )
    plot_values = np.asarray(plot_arrays.pop("values"), dtype=np.float64).reshape(-1)
    extras_plot = plot_arrays

    if mode == "zero_outside":
        # Display the fill value outside the tumor and derive color limits from tumor patches.
        plot_clip_lo, plot_clip_hi = equivalent_clip_percentiles(
            plot_values, tumor_values, clip_lo, clip_hi
        )
        if log:
            print(
                f"[region] Tumor region with zero outside: "
                f"tumor={tumor_values.size} / total={plot_values.size}"
            )
            print(
                f"[region] Tumor patch color limits use "
                f"P{float(clip_lo):g}/P{float(clip_hi):g}，"
                f" and correspond to full-tissue limits of "
                f"P{plot_clip_lo:.4f}/P{plot_clip_hi:.4f}"
            )
    else:
        plot_clip_lo, plot_clip_hi = float(clip_lo), float(clip_hi)
        if log:
            print(
                f"[region] Tumor patches only: "
                f"patches={tumor_values.size}"
            )

    return TumorDisplayResult(
        plot_coords=plot_coords,
        plot_values=plot_values,
        tumor_coords=tumor_coords,
        tumor_values=tumor_values,
        tumor_indices=np.asarray(tumor_indices, dtype=np.int64).reshape(-1),
        clip_lo=plot_clip_lo,
        clip_hi=plot_clip_hi,
        extras_plot=extras_plot,
        extras_tumor=extras_tumor_arr,
        filter_info=info,
        display_mode=mode,
    )


def prepare_tumor_display(
    coords: np.ndarray,
    values: np.ndarray,
    case_id: str,
    *,
    extras: Optional[Dict[str, object]] = None,
    display_mode: str = DEFAULT_TUMOR_DISPLAY_MODE,
    clip_lo: float = 0.5,
    clip_hi: float = 99.5,
    value_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fill: float = 0.0,
    annotation_csv: Optional[PathLike] = None,
    annotation_dir: Optional[PathLike] = None,
    enabled: bool = True,
    require: bool = True,
    empty_ok: bool = False,
    log: bool = True,
) -> TumorDisplayResult:
    """Identify tumor tiles using annotation CSVs and build arrays for the selected display mode."""
    all_coords = np.asarray(coords)
    all_values = np.asarray(values, dtype=np.float64).reshape(-1)
    extras = dict(extras or {})
    all_indices = np.arange(all_values.shape[0], dtype=np.int64)
    arrays = {"values": all_values, "indices": all_indices, **extras}
    tumor_coords, filtered, info = apply_tumor_filter(
        all_coords,
        case_id,
        arrays=arrays,
        annotation_csv=annotation_csv,
        annotation_dir=annotation_dir,
        enabled=enabled,
        require=require,
        empty_ok=empty_ok,
        log=log,
    )
    tumor_values = np.asarray(filtered["values"], dtype=np.float64).reshape(-1)
    tumor_indices = np.asarray(filtered["indices"], dtype=np.int64).reshape(-1)
    extras_tumor = {key: filtered[key] for key in extras}
    if value_transform is None:
        display_values = tumor_values
    else:
        display_values = np.asarray(
            value_transform(tumor_values), dtype=np.float64
        ).reshape(-1)
    return compose_tumor_display(
        all_coords,
        tumor_coords,
        tumor_indices,
        display_values,
        enabled=bool(info["enabled"]),
        display_mode=display_mode,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        extras_tumor=extras_tumor,
        fill=fill,
        filter_info=info,
        log=log,
    )


def add_tumor_display_mode_argument(
    parser,
    *,
    default: str = DEFAULT_TUMOR_DISPLAY_MODE,
) -> None:
    """Add the tumor display mode option to the parser."""
    parser.add_argument(
        "--tumor-display-mode",
        choices=list(TUMOR_DISPLAY_MODES),
        default=_normalize_display_mode(default),
        help=(
            'Display mode after tumor filtering. '
            'filter plots tumor patches only. '
            'zero_outside retains all tissue and displays other patches as zero.'
        ),
    )


def add_tumor_region_arguments(
    parser,
    *,
    single_csv: bool = True,
    annotation_dir_default: Optional[PathLike] = None,
    annotation_csv_default: Optional[PathLike] = None,
    tumor_filter_default: bool = DEFAULT_TUMOR_FILTER,
    display_mode_default: Optional[str] = None,
) -> None:
    """Add annotation inputs and the option to disable tumor filtering."""
    default_dir = (
        DEFAULT_ANNOTATION_DIR
        if annotation_dir_default is None
        else Path(annotation_dir_default)
    )
    group = parser.add_argument_group('Tumor annotation filtering')
    group.add_argument(
        "--annotation-dir",
        type=Path,
        default=default_dir,
        help='Directory containing per-slide patch annotation CSVs',
    )
    if single_csv:
        csv_default = DEFAULT_ANNOTATION_CSV if annotation_csv_default is None else Path(
            annotation_csv_default
        )
        group.add_argument(
            "--annotation-csv",
            type=Path,
            default=csv_default,
            help='Annotation CSV for one slide. Takes precedence over directory lookup.',
        )
    group.add_argument(
        "--tumor-filter",
        dest="tumor_filter",
        action="store_true",
        help='Process patches inside tumor annotations',
    )
    group.add_argument(
        "--no-tumor-filter",
        dest="tumor_filter",
        action="store_false",
        help='Disable tumor filtering and process all patches',
    )
    # Set the parser default explicitly when multiple options share the same destination.
    parser.set_defaults(tumor_filter=bool(tumor_filter_default))
    if display_mode_default is not None:
        add_tumor_display_mode_argument(
            parser,
            default=display_mode_default,
        )


def tumor_filter_kwargs(args) -> dict:
    """Extract the tumor filter options from the argument namespace."""
    return {
        "annotation_csv": getattr(args, "annotation_csv", None),
        "annotation_dir": getattr(args, "annotation_dir", None),
        "enabled": bool(getattr(args, "tumor_filter", DEFAULT_TUMOR_FILTER)),
    }
