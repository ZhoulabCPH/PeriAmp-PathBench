"""Shared CSV and path parsing, job setup, and NPZ helpers for attention and attribution scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import numpy as np

SUPPORTED_SLIDE_EXTS = (
    ".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".svslide", ".bif", ".scn",
)

# Batch output subdirectories
CATEGORY_ATTR_NPZ = "attributions"
CATEGORY_ATTN_NPZ = "attentions"

_GENERIC_NPZ_STEMS = frozenset({"attributions", "attention", "attentions"})


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def sanitize(name: str) -> str:
    """Replace characters that are invalid in cross-platform filenames."""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in str(name))
    return out.strip().strip(".") or "unnamed"


def extract_mil_name(config_label: str, feature_label: Optional[str]) -> str:
    """Remove the encoder label from the configuration label to obtain the MIL name."""
    if not feature_label:
        return config_label
    prefix = feature_label + "_"
    if config_label.startswith(prefix):
        return config_label[len(prefix):]
    return config_label


def category_path(outroot: Path, category: str, prefix: str, ext: str) -> Path:
    """<outroot>/<category>/<prefix>-<category>.<ext>"""
    subdir = outroot / category
    subdir.mkdir(parents=True, exist_ok=True)
    ext = ext.lstrip(".")
    return subdir / f"{prefix}-{category}.{ext}"


def category_path_pdf(outroot: Path, category: str, prefix: str) -> Path:
    """<outroot>/pdf/<category>/<prefix>-<category>.pdf"""
    return category_path(outroot / "pdf", category, prefix, "pdf")


def collect_h5_files(feature_dir: Path) -> List[Path]:
    return sorted(feature_dir.glob("*.h5"))


def find_slide_for_stem(slide_dir: Optional[Path], stem: str) -> Optional[Path]:
    if slide_dir is None:
        return None
    for ext in SUPPORTED_SLIDE_EXTS:
        path = slide_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def iter_h5(h5_input: Path) -> List[Path]:
    if h5_input.is_file():
        return [h5_input]
    if h5_input.is_dir():
        return sorted(h5_input.glob("*.h5"))
    raise FileNotFoundError(f"Invalid HDF5 path: {h5_input}")


def iter_npz(npz_input: Path, *, recursive: bool = True) -> List[Path]:
    if npz_input.is_file():
        return [npz_input]
    if npz_input.is_dir():
        if recursive:
            return sorted(npz_input.rglob("*.npz"))
        return sorted(npz_input.glob("*.npz"))
    raise FileNotFoundError(f"Invalid NPZ path: {npz_input}")


def npz_scalar(data: dict, key: str, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    return value[0].item() if value.size else default


def npz_text_list(value) -> List[str]:
    if value is None:
        return []
    result: List[str] = []
    for item in np.asarray(value).reshape(-1).tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        result.append(str(item))
    return result


def infer_case_id_from_npz(npz_path: Path, data: dict) -> str:
    if "case_id" in data:
        value = np.asarray(data["case_id"]).reshape(-1)
        if value.size:
            item = value[0]
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            return str(item)
    if npz_path.stem in _GENERIC_NPZ_STEMS:
        return npz_path.parent.name
    return npz_path.stem.split("-")[0]


def output_name_from_npz(npz_path: Path, case_id: str) -> str:
    if npz_path.stem in _GENERIC_NPZ_STEMS:
        return case_id
    return npz_path.stem


def _parse_two_col_csv(
    csv_path: Path,
    path_keys: List[str],
    label_key: str = "label",
) -> List[dict]:
    entries: List[dict] = []
    path_keys_lc = [k.lower() for k in path_keys]
    label_key_lc = label_key.lower()

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]

    if not rows:
        return entries

    first_cell = rows[0][0].strip().lower() if rows[0] else ""
    has_header = first_cell in path_keys_lc

    if has_header:
        header = [c.strip().lower() for c in rows[0]]
        path_idx = -1
        for key in path_keys_lc:
            if key in header:
                path_idx = header.index(key)
                break
        if path_idx < 0:
            raise ValueError(f"CSV does not contain the expected path column {path_keys} (found: {rows[0]})")
        label_idx = header.index(label_key_lc) if label_key_lc in header else -1

        for row in rows[1:]:
            if not row or path_idx >= len(row):
                continue
            path_val = row[path_idx].strip()
            if not path_val or path_val.startswith("#"):
                continue
            label: Optional[str] = None
            if 0 <= label_idx < len(row):
                label = row[label_idx].strip() or None
            entries.append({"path": path_val, "label": label})
    else:
        for row in rows:
            path_val = row[0].strip()
            if not path_val or path_val.startswith("#"):
                continue
            label = row[1].strip() if len(row) > 1 and row[1].strip() else None
            entries.append({"path": path_val, "label": label})
    return entries


def parse_config_csv(csv_path: Path) -> List[dict]:
    raw = _parse_two_col_csv(csv_path, path_keys=["config_path", "config"])
    entries: List[dict] = []
    for entry in raw:
        label = entry["label"] or Path(entry["path"]).parent.name
        entries.append({"config_path": entry["path"], "label": label})
    return entries


def parse_feature_csv(csv_path: Path) -> List[dict]:
    raw = _parse_two_col_csv(csv_path, path_keys=["feature_path", "feature", "path"])
    entries: List[dict] = []
    for entry in raw:
        feat_dir = Path(entry["path"])
        if not feat_dir.is_dir():
            raise NotADirectoryError(
                f"[feature CSV] Not a feature directory: {feat_dir}"
            )
        label = entry["label"] or feat_dir.name
        h5_list = collect_h5_files(feat_dir)
        if not h5_list:
            print(f"[warning] No HDF5 files found in {feat_dir} (label={label})")
        entries.append({
            "feature_path": str(feat_dir),
            "label": label,
            "h5_files": h5_list,
        })
    return entries


def match_feature_for_config(
    feature_entries: List[dict],
    config_label: str,
) -> Optional[dict]:
    best: Optional[dict] = None
    for fe in feature_entries:
        fl = fe["label"]
        if config_label == fl or config_label.startswith(fl + "_"):
            if best is None or len(fe["label"]) > len(best["label"]):
                best = fe
    return best
