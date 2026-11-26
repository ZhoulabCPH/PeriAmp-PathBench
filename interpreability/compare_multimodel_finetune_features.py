"""
Compare feature changes after fine-tuning across pathology encoders.
Compare raw features only within each encoder. Fit sampled PCA and RGB scaling separately for each encoder. Compare encoder pairs using standardized change maps, top-k overlap, and linear CKA.
The --csv-only option skips PCA fitting, colormap estimation, and image output to reduce memory use.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.stats import rankdata, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from PCA_visulize_feature import (
    load_h5,
    sort_by_coords,
    validate_inputs,
    robust_percentile_limits,
    make_rgb_canvas_from_coords,
    make_scalar_canvas_from_coords,
    save_rgb_canvas,
    save_scalar_map,
    build_rgb_overlay,
    build_scalar_overlay,
    collect_files_by_stem,
    unique_path_list,
    normalize_rgb_with_shared_scaler,
    openslide,
    save_overlay_outputs,
)
# =====================================================================
# Command-line defaults
# Keep argument defaults together for inspection and reproduction.
# Set explicit options to override the defaults for an experiment.
# =====================================================================
DEFAULT_TILE_SIZE = 256
DEFAULT_DOWNSAMPLE = 16
DEFAULT_ALPHA = 0.55
DEFAULT_SMOOTH_SIGMA = 2.0
DEFAULT_DARKEN_BG = 0.8

DEFAULT_PCA_SAMPLE = 50000

DEFAULT_ENABLE_GEOMETRY_CHANGE = False
DEFAULT_NUM_LANDMARKS = 2000
DEFAULT_GEOMETRY_BATCH_SIZE = 4096

DEFAULT_TOP_K_PERCENTS = [1.0, 5.0, 10.0]

# Write CSV metrics only when enabled.
DEFAULT_CSV_ONLY = True

# =====================================================================
# Skip feature sources without valid bags.
# =====================================================================
def is_empty_features(features):
    """Check whether the feature collection contains any usable features."""
    if features is None:
        return True
    arr = np.asarray(features)
    if arr.size == 0 or arr.ndim == 0 or arr.shape[0] == 0:
        return True
    return False


def safe_load_h5(h5_path):
    """Read features and coords from an HDF5 file. Return None if the file is unreadable, empty, or missing either dataset."""
    try:
        features, coords = load_h5(str(h5_path))
    except Exception as e:
        print(f"  [Skip] cannot read features from {h5_path}: {e}")
        return None
    if is_empty_features(features):
        print(f"  [Skip] empty features (treated as absent): {h5_path}")
        return None
    return features, coords


# =====================================================================
# Sample HDF5 features for per-model PCA and color limits.
# =====================================================================
# Default cap on the number of sampled tiles.
# A bounded sample avoids exhausting memory on large slides.

def resolve_sample_budget(sample):
    """Use the bounded default when the requested sample budget is nonpositive."""
    if sample is None or sample <= 0:
        return DEFAULT_PCA_SAMPLE
    return int(sample)


def _read_sampled_rows(h5_path, n_sample, rng):
    """Read at most n_sample random rows using sorted, unique HDF5 indices. Return float32 features or None if the input is unavailable."""
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "features" not in f:
                return None
            dset = f["features"]
            if dset.ndim != 2 or dset.shape[0] == 0:
                return None
            n = dset.shape[0]
            if n_sample >= n:
                feats = dset[:]
            else:
                idx = np.sort(rng.choice(n, size=n_sample, replace=False))
                feats = dset[idx, :]
    except Exception as e:
        print(f"  [Skip] cannot sample features from {h5_path}: {e}")
        return None
    return np.asarray(feats, dtype=np.float32)


def fit_model_pca_and_rgb_scaler_sampled(h5_paths, sample_budget=DEFAULT_PCA_SAMPLE,
                                         random_state=42):
    """Fit PCA and a clipped RGB min-max scaler to sampled before and after features from one encoder. Divide the sampling budget across files to bound memory use."""
    sample_budget = resolve_sample_budget(sample_budget)

    rng = np.random.default_rng(random_state)
    num_files = max(1, len(h5_paths))
    per_file = max(1, sample_budget // num_files)

    fit_arrays = []
    feature_dim = None
    for p in h5_paths:
        feats = _read_sampled_rows(p, per_file, rng)
        if feats is None or feats.size == 0:
            continue
        if feature_dim is None:
            feature_dim = feats.shape[1]
        elif feats.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dim mismatch in model PCA: {p} has dim {feats.shape[1]}, "
                f"expected {feature_dim}"
            )
        fit_arrays.append(feats)

    if not fit_arrays:
        raise RuntimeError("No features available to fit model PCA (all empty/unreadable?).")

    fit_features = np.concatenate(fit_arrays, axis=0)
    del fit_arrays

    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(fit_features)

    sampled_pca = pca.transform(fit_features)
    del fit_features

    rgb_scaler = MinMaxScaler(clip=True)
    rgb_scaler.fit(sampled_pca)

    return pca, rgb_scaler


def estimate_delta_cosine_limits_sampled(jobs, sample_budget=DEFAULT_PCA_SAMPLE,
                                         random_state=42):
    """Estimate shared color limits for delta and cosine maps within one encoder. Read one before/after pair at a time and retain only sampled scalar values."""
    sample_budget = resolve_sample_budget(sample_budget)

    rng = np.random.default_rng(random_state)
    num_jobs = max(1, len(jobs))
    per_job = max(1, sample_budget // num_jobs)

    all_delta = []
    all_cosine = []
    for job in jobs:
        before_loaded = safe_load_h5(job["before_h5"])
        after_loaded = safe_load_h5(job["after_h5"])
        if before_loaded is None or after_loaded is None:
            continue

        bf, _ = sort_by_coords(*before_loaded)
        af, _ = sort_by_coords(*after_loaded)

        delta = np.linalg.norm(af - bf, axis=1)
        bn = np.linalg.norm(bf, axis=1)
        an = np.linalg.norm(af, axis=1)
        cosine = np.sum(bf * af, axis=1) / np.maximum(bn * an, 1e-12)

        del bf, af, before_loaded, after_loaded  # Release the job's feature arrays.

        n = len(delta)
        if n > per_job:
            idx = rng.choice(n, size=per_job, replace=False)
            delta = delta[idx]
            cosine = cosine[idx]

        all_delta.append(delta.astype(np.float32, copy=False))
        all_cosine.append(cosine.astype(np.float32, copy=False))

    if not all_delta:
        return (0.0, 1.0), (0.0, 1.0)

    delta_limits = robust_percentile_limits(np.concatenate(all_delta), low=1, high=99)
    cosine_limits = robust_percentile_limits(np.concatenate(all_cosine), low=2, high=98)
    return delta_limits, cosine_limits


# =====================================================================
# Scalar normalization for cross-model comparison
# =====================================================================
def percentile_normalize(values, low=1, high=99):
    """Scale finite values from the supplied percentile bounds to [0, 1]. Use zero for nonfinite values and return float32."""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float32)

    vmin, vmax = np.percentile(values[finite_mask], [low, high])
    normalized = (values - vmin) / (vmax - vmin + 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite_mask] = 0.0
    return normalized.astype(np.float32)


def rank_normalize(values):
    """Map finite values to ranks in [0, 1] for scale-independent comparison. Use zero for nonfinite values and return float32."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros(n, dtype=np.float32)

    # Place nonfinite values at the lowest rank.
    safe = np.where(finite_mask, values, -np.inf)
    ranks = rankdata(safe, method="average")
    out = (ranks - 1) / (n - 1 + 1e-12)
    out[~finite_mask] = 0.0
    return out.astype(np.float32)


# =====================================================================
# Linear CKA for representational similarity
# =====================================================================
def center_features(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """Compute linear centered kernel alignment for row-aligned matrices with equal sample counts and possibly different feature widths."""
    X = center_features(np.asarray(X, dtype=np.float64))
    Y = center_features(np.asarray(Y, dtype=np.float64))

    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")

    return float(xy / (xx * yy + 1e-12))


# =====================================================================
# Optional landmark-based local geometry comparison
# =====================================================================
def local_geometry_change_with_landmarks(
        before, after, num_landmarks=2000, random_state=42, batch_size=4096):
    """Compare each matched tile's similarities to landmark tiles before and after fine-tuning. Process row batches to avoid a full pairwise matrix."""
    rng = np.random.default_rng(random_state)
    n = len(before)

    landmark_idx = rng.choice(n, size=min(num_landmarks, n), replace=False)

    # Normalize in float32 to reduce memory use.
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)

    before_norm = before / (np.linalg.norm(before, axis=1, keepdims=True) + 1e-12)
    after_norm = after / (np.linalg.norm(after, axis=1, keepdims=True) + 1e-12)

    before_landmarks = before_norm[landmark_idx]
    after_landmarks = after_norm[landmark_idx]

    geo_change = np.empty(n, dtype=np.float32)
    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        end = min(start + bs, n)
        sim_before = before_norm[start:end] @ before_landmarks.T
        sim_after = after_norm[start:end] @ after_landmarks.T
        geo_change[start:end] = np.linalg.norm(sim_after - sim_before, axis=1)

    return geo_change


# =====================================================================
# Overlap of the highest-change tiles
# =====================================================================
def top_k_indices(values, top_k_percent):
    """Return the indices of the largest finite values at the requested percentage."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return set()

    k = max(1, int(round(n * top_k_percent / 100.0)))
    k = min(k, n)

    finite_mask = np.isfinite(values)
    safe = np.where(finite_mask, values, -np.inf)
    idx = np.argsort(safe, kind="stable")[::-1][:k]
    idx = [int(i) for i in idx if finite_mask[i]]
    return set(idx)


def jaccard_dice(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = inter / union if union > 0 else 0.0
    dice = 2 * inter / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0.0
    return inter, jaccard, dice


# =====================================================================
# Scalar maps and optional overlays
# =====================================================================
def save_scalar_map_and_overlay(
        values,
        coords,
        out_map_path,
        args,
        slide_path=None,
        cmap="magma",
        vmin=None,
        vmax=None,
        title="",
        overlay_path=None,
):
    """Save a scalar canvas as PNG and add an overlay when a slide path is provided."""
    canvas = make_scalar_canvas_from_coords(
        values=values,
        coords=coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
    )
    save_scalar_map(
        canvas,
        out_map_path,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        title=title,
    )

    if overlay_path is not None and slide_path is not None and openslide is not None:
        mask = np.isfinite(canvas)
        # Use the coordinate origin expected by the scalar canvas.
        coords_i = coords.astype(np.int64)
        origin_xy = (coords_i[:, 0].min(), coords_i[:, 1].min())
        try:
            layers = build_scalar_overlay(
                slide_path=str(slide_path),
                scalar_canvas=canvas,
                mask=mask,
                origin_xy=origin_xy,
                downsample=args.downsample,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=args.alpha,
                smooth_sigma=args.smooth_sigma,
                darken_bg=args.darken_bg,
            )
            save_overlay_outputs(
                layers,
                overlay_path,
                Path(overlay_path).with_suffix(".pdf"),
                alpha=args.alpha,
            )
        except Exception as e:
            print(f"    [Warning] scalar overlay failed for {out_map_path.name}: {e}")

    return canvas


# =====================================================================
# Process one case and checkpoint within one model.
# =====================================================================
def process_model_case(
        model_name,
        case_name,
        checkpoint_name,
        before_h5,
        after_h5,
        slide_path,
        args,
        out_ckpt_dir,
        model_pca,
        model_rgb_scaler,
        delta_limits,
        cosine_limits,
):
    """Produce PCA, feature delta, cosine, and optional geometry-change maps for a before/after pair. In CSV-only mode compute scalar values and metrics without images. Return normalized scalars, coordinates, and CKA records for cross-model comparison."""
    csv_only = bool(getattr(args, "csv_only", False))
    print(f"  [{model_name}/{checkpoint_name}] case={case_name}"
          + (" [csv-only]" if csv_only else ""))

    before_features, before_coords = sort_by_coords(*load_h5(str(before_h5)))
    after_features, after_coords = sort_by_coords(*load_h5(str(after_h5)))

    # Require matching coordinates, tile counts, and feature widths within a model.
    validate_inputs(before_features, before_coords, after_features, after_coords)

    coords = before_coords
    num_tiles = len(before_features)
    feature_dim = before_features.shape[1]

    # Fit and plot each model's PCA unless CSV-only mode is active.
    if not csv_only:
        before_pca = model_pca.transform(before_features)
        after_pca = model_pca.transform(after_features)

        before_rgb = normalize_rgb_with_shared_scaler(before_pca, model_rgb_scaler)
        after_rgb = normalize_rgb_with_shared_scaler(after_pca, model_rgb_scaler)

        before_canvas, before_mask, origin_xy, _ = make_rgb_canvas_from_coords(
            rgb=before_rgb, coords=coords,
            tile_size=args.tile_size, downsample=args.downsample,
            background_color=(1.0, 1.0, 1.0),
        )
        save_rgb_canvas(before_canvas, out_ckpt_dir / f"{case_name}_before_pca_map.png")

        after_canvas, after_mask, _, _ = make_rgb_canvas_from_coords(
            rgb=after_rgb, coords=coords,
            tile_size=args.tile_size, downsample=args.downsample,
            background_color=(1.0, 1.0, 1.0),
        )
        save_rgb_canvas(after_canvas, out_ckpt_dir / f"{case_name}_after_pca_map.png")

    # ---- delta / cosine ----
    delta = np.linalg.norm(after_features - before_features, axis=1)

    before_norm = np.linalg.norm(before_features, axis=1)
    after_norm = np.linalg.norm(after_features, axis=1)
    denom = np.maximum(before_norm * after_norm, 1e-12)
    cosine_sim = np.sum(before_features * after_features, axis=1) / denom
    cosine_change = 1.0 - cosine_sim

    # Normalize change maps for cross-model comparison.
    delta_percentile_norm = percentile_normalize(delta, low=1, high=99)
    delta_rank_norm = rank_normalize(delta)
    cosine_change_percentile_norm = percentile_normalize(cosine_change, low=1, high=99)
    cosine_change_rank_norm = rank_normalize(cosine_change)

    # Write scalar maps and overlays when plotting is enabled.
    if not csv_only:
        delta_vmin, delta_vmax = delta_limits
        cos_vmin, cos_vmax = cosine_limits
        # Derive the cosine-change limits from the shared cosine-similarity limits.
        cc_vmin = 1.0 - cos_vmax
        cc_vmax = 1.0 - cos_vmin

        save_scalar_map_and_overlay(
            delta, coords, out_ckpt_dir / f"{case_name}_delta_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=delta_vmin, vmax=delta_vmax,
            title=f"{model_name}/{checkpoint_name} {case_name} - Delta (raw)",
            overlay_path=out_ckpt_dir / f"{case_name}_delta_overlay.png",
        )
        save_scalar_map_and_overlay(
            delta_percentile_norm, coords,
            out_ckpt_dir / f"{case_name}_delta_percentile_norm_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
            title=f"{model_name}/{checkpoint_name} {case_name} - Delta (percentile norm)",
        )
        save_scalar_map_and_overlay(
            delta_rank_norm, coords,
            out_ckpt_dir / f"{case_name}_delta_rank_norm_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
            title=f"{model_name}/{checkpoint_name} {case_name} - Delta (rank norm)",
        )
        save_scalar_map_and_overlay(
            cosine_sim, coords,
            out_ckpt_dir / f"{case_name}_cosine_similarity_map.png", args,
            slide_path=slide_path, cmap="viridis", vmin=cos_vmin, vmax=cos_vmax,
            title=f"{model_name}/{checkpoint_name} {case_name} - Cosine Similarity",
        )
        save_scalar_map_and_overlay(
            cosine_change, coords,
            out_ckpt_dir / f"{case_name}_cosine_change_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=cc_vmin, vmax=cc_vmax,
            title=f"{model_name}/{checkpoint_name} {case_name} - Cosine Change",
        )
        save_scalar_map_and_overlay(
            cosine_change_percentile_norm, coords,
            out_ckpt_dir / f"{case_name}_cosine_change_percentile_norm_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
            title=f"{model_name}/{checkpoint_name} {case_name} - Cosine Change (percentile norm)",
        )
        save_scalar_map_and_overlay(
            cosine_change_rank_norm, coords,
            out_ckpt_dir / f"{case_name}_cosine_change_rank_norm_map.png", args,
            slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
            title=f"{model_name}/{checkpoint_name} {case_name} - Cosine Change (rank norm)",
        )

        # Optional PCA overlays
        if slide_path is not None and openslide is not None:
            try:
                before_layers = build_rgb_overlay(
                    slide_path=str(slide_path), pca_canvas=before_canvas, mask=before_mask,
                    origin_xy=origin_xy, downsample=args.downsample,
                    alpha=args.alpha, smooth_sigma=args.smooth_sigma, darken_bg=args.darken_bg,
                )
                after_layers = build_rgb_overlay(
                    slide_path=str(slide_path), pca_canvas=after_canvas, mask=after_mask,
                    origin_xy=origin_xy, downsample=args.downsample,
                    alpha=args.alpha, smooth_sigma=args.smooth_sigma, darken_bg=args.darken_bg,
                )
                save_overlay_outputs(
                    before_layers,
                    out_ckpt_dir / f"{case_name}_before_pca_overlay.png",
                    alpha=args.alpha,
                )
                save_overlay_outputs(
                    after_layers,
                    out_ckpt_dir / f"{case_name}_after_pca_overlay.png",
                    alpha=args.alpha,
                )
            except Exception as e:
                print(f"    [Warning] PCA overlay failed for {case_name}: {e}")

    # Optional local geometry change
    geometry_percentile_norm = None
    geometry_rank_norm = None
    if args.enable_geometry_change:
        geo_change = local_geometry_change_with_landmarks(
            before_features, after_features,
            num_landmarks=args.num_landmarks, random_state=42,
            batch_size=args.geometry_batch_size,
        )
        geometry_percentile_norm = percentile_normalize(geo_change, low=1, high=99)
        geometry_rank_norm = rank_normalize(geo_change)

        if not csv_only:
            save_scalar_map_and_overlay(
                geo_change, coords,
                out_ckpt_dir / f"{case_name}_geometry_change_map.png", args,
                slide_path=slide_path, cmap="magma", vmin=None, vmax=None,
                title=f"{model_name}/{checkpoint_name} {case_name} - Geometry Change",
            )
            save_scalar_map_and_overlay(
                geometry_percentile_norm, coords,
                out_ckpt_dir / f"{case_name}_geometry_change_percentile_norm_map.png", args,
                slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
                title=f"{model_name}/{checkpoint_name} {case_name} - Geometry Change (percentile norm)",
            )
            save_scalar_map_and_overlay(
                geometry_rank_norm, coords,
                out_ckpt_dir / f"{case_name}_geometry_change_rank_norm_map.png", args,
                slide_path=slide_path, cmap="magma", vmin=0.0, vmax=1.0,
                title=f"{model_name}/{checkpoint_name} {case_name} - Geometry Change (rank norm)",
            )

    # ---- CKA ----
    cka_before_after = linear_cka(before_features, after_features)
    cka_change_score = 1.0 - cka_before_after

    # Per-model summary record
    summary_row = {
        "model_name": model_name,
        "case_name": case_name,
        "checkpoint_name": checkpoint_name,
        "num_tiles": num_tiles,
        "feature_dim": feature_dim,
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "cosine_sim_mean": float(np.mean(cosine_sim)),
        "cosine_change_mean": float(np.mean(cosine_change)),
        "cka_before_after": cka_before_after,
        "cka_change_score": cka_change_score,
    }

    cka_row = {
        "model_name": model_name,
        "case_name": case_name,
        "checkpoint_name": checkpoint_name,
        "cka_before_after": cka_before_after,
        "cka_change_score": cka_change_score,
        "num_tiles": num_tiles,
        "feature_dim": feature_dim,
    }

    # Standardized maps for cross-model analysis
    # Keep paths for on-demand CKA loading and retain normalized maps as float32.
    cross_model_maps = {
        "coords": coords,
        "num_tiles": num_tiles,
        "feature_dim": feature_dim,
        "before_h5": before_h5,
        "after_h5": after_h5,
        "delta_percentile_norm": delta_percentile_norm,
        "delta_rank_norm": delta_rank_norm,
        "cosine_change_percentile_norm": cosine_change_percentile_norm,
        "cosine_change_rank_norm": cosine_change_rank_norm,
    }
    if geometry_rank_norm is not None:
        cross_model_maps["geometry_change_percentile_norm"] = geometry_percentile_norm
        cross_model_maps["geometry_change_rank_norm"] = geometry_rank_norm

    return summary_row, cka_row, cross_model_maps


# =====================================================================
# Process one encoder.
# =====================================================================
def process_one_model(model_cfg, slide_files, slide_dir_exists, args, per_model_root):
    """Collect valid case and checkpoint pairs, fit model-specific PCA and color limits, and write summary_metrics.csv. Return scalar data and CKA rows for cross-model comparison."""
    model_name = model_cfg["name"]
    before_dir = Path(model_cfg["before_dir"])
    after_dirs = model_cfg["after_dirs"]  # dict: checkpoint_name -> dir

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print(f"before_dir: {before_dir}")
    print("=" * 60)

    model_root = per_model_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)

    if not before_dir.exists():
        print(f"[Error] before_dir does not exist for {model_name}: {before_dir}. Skip model.")
        return {}, []

    before_files = collect_files_by_stem(before_dir, [".h5"])
    if not before_files:
        print(f"[Error] No h5 in before_dir for {model_name}: {before_dir}. Skip model.")
        return {}, []

    # Collect valid before and after jobs.
    jobs = []
    for checkpoint_name, after_dir in after_dirs.items():
        after_dir = Path(after_dir)
        if not after_dir.exists():
            print(f"[Warning] after_dir missing: {after_dir} ({model_name}/{checkpoint_name})")
            continue

        after_files = collect_files_by_stem(after_dir, [".h5"])
        common = sorted(set(before_files.keys()) & set(after_files.keys()))
        if not common:
            print(f"[Warning] No matched cases for {model_name}/{checkpoint_name} ({after_dir})")
            continue

        for case_name in common:
            jobs.append({
                "case_name": case_name,
                "checkpoint_name": checkpoint_name,
                "before_h5": before_files[case_name],
                "after_h5": after_files[case_name],
                "slide_path": slide_files.get(case_name, None),
            })

    if not jobs:
        print(f"[Error] No matched before-after jobs for {model_name}. Skip model.")
        return {}, []

    # Validate shape and coordinate alignment and skip empty features.
    valid_jobs = []
    for job in jobs:
        # Skip jobs with missing or empty before or after features.
        before_loaded = safe_load_h5(job["before_h5"])
        after_loaded = safe_load_h5(job["after_h5"])
        if before_loaded is None or after_loaded is None:
            print(f"  [Skip] {model_name}/{job['checkpoint_name']} {job['case_name']}: "
                  f"empty/missing features, related maps skipped.")
            continue

        try:
            bf, bc = sort_by_coords(*before_loaded)
            af, ac = sort_by_coords(*after_loaded)
            validate_inputs(bf, bc, af, ac)
            valid_jobs.append(job)
        except Exception as e:
            print(f"  [Error] invalid job skipped: {model_name}/{job['checkpoint_name']} "
                  f"{job['case_name']}: {e}")
    jobs = valid_jobs
    if not jobs:
        print(f"[Error] No valid jobs after validation for {model_name}. Skip model.")
        return {}, []

    csv_only = bool(getattr(args, "csv_only", False))
    model_pca = None
    model_rgb_scaler = None
    delta_limits = (None, None)
    cosine_limits = (None, None)

    if csv_only:
        # Skip plot-only PCA and color limits in CSV-only mode.
        print(f"  [csv-only] skip model-specific PCA fitting and color-limit estimation.")
    else:
        # Fit PCA using only this encoder's before and after features.
        feature_paths = []
        for job in jobs:
            feature_paths.append(job["before_h5"])
            feature_paths.append(job["after_h5"])
        feature_paths = unique_path_list(feature_paths)

        sample_budget = resolve_sample_budget(args.sample)
        print(f"  Fitting model-specific PCA on {len(feature_paths)} unique H5 "
              f"(sampled up to {sample_budget} rows total) ...")
        model_pca, model_rgb_scaler = fit_model_pca_and_rgb_scaler_sampled(
            h5_paths=feature_paths, sample_budget=sample_budget, random_state=42,
        )
        print(f"  {model_name} PCA explained variance ratio: {model_pca.explained_variance_ratio_}")

        # Estimate shared delta and cosine color limits from sampled tiles.
        delta_limits, cosine_limits = estimate_delta_cosine_limits_sampled(
            jobs, sample_budget=sample_budget, random_state=42,
        )
        print(f"  {model_name} delta limits : {delta_limits}")
        print(f"  {model_name} cosine limits: {cosine_limits}")

    # Process each comparison.
    summary_rows = []
    cka_rows = []
    cross_model_store = {}  # (case, checkpoint) -> {model_name: cross_model_maps}

    for job in jobs:
        checkpoint_name = job["checkpoint_name"]
        out_ckpt_dir = model_root / checkpoint_name
        # Create the output directory for downstream paths even in CSV-only mode.
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)

        if (not csv_only) and slide_dir_exists and job["slide_path"] is None:
            print(f"  [Warning] no WSI for {job['case_name']}, maps only (no overlay).")

        try:
            summary_row, cka_row, cm_maps = process_model_case(
                model_name=model_name,
                case_name=job["case_name"],
                checkpoint_name=checkpoint_name,
                before_h5=job["before_h5"],
                after_h5=job["after_h5"],
                slide_path=job["slide_path"],
                args=args,
                out_ckpt_dir=out_ckpt_dir,
                model_pca=model_pca,
                model_rgb_scaler=model_rgb_scaler,
                delta_limits=delta_limits,
                cosine_limits=cosine_limits,
            )
            summary_rows.append(summary_row)
            cka_rows.append(cka_row)

            key = (job["case_name"], checkpoint_name)
            cross_model_store.setdefault(key, {})[model_name] = cm_maps
        except Exception as e:
            print(f"  [Error] failed {model_name}/{checkpoint_name} {job['case_name']}: {e}")

    # Write the model's summary_metrics.csv.
    if summary_rows:
        write_csv(model_root / "summary_metrics.csv", summary_rows)
        print(f"  Wrote {model_root / 'summary_metrics.csv'}")

    return cross_model_store, cka_rows


# =====================================================================
# CSV output helpers
# =====================================================================
def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# Cross-model correlations and top-k overlap
# =====================================================================
# Map each normalized output to its comparison category.
DELTA_METRICS = ["delta_percentile_norm", "delta_rank_norm"]
COSINE_METRICS = ["cosine_change_percentile_norm", "cosine_change_rank_norm"]
GEOMETRY_METRICS = ["geometry_change_percentile_norm", "geometry_change_rank_norm"]

# Use rank-normalized maps for top-k overlap.
TOPK_METRICS = ["delta_rank_norm", "cosine_change_rank_norm", "geometry_change_rank_norm"]


def safe_corr(a, b):
    """Compute Pearson or Spearman correlation on finite pairs. Return NaN for insufficient samples or constant vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan, np.nan

    a = a[mask]
    b = b[mask]

    # A constant vector has undefined correlation.
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan

    try:
        pear = pearsonr(a, b)[0]
    except Exception:
        pear = np.nan
    try:
        spear = spearmanr(a, b)[0]
    except Exception:
        spear = np.nan
    return float(pear), float(spear)


def cross_model_analysis(cross_model_store, out_dir, args):
    """Compare standardized maps across models for each case and checkpoint when tile coordinates match. Save correlations and top-k overlaps."""
    delta_corr_rows = []
    cosine_corr_rows = []
    geometry_corr_rows = []
    topk_rows = []

    top_k_percents = [float(x) for x in args.top_k_percents]

    for (case_name, checkpoint_name), model_dict in sorted(cross_model_store.items()):
        model_names = sorted(model_dict.keys())
        if len(model_names) < 2:
            continue  # Pairwise comparison requires at least two models.

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a = model_names[i]
                model_b = model_names[j]
                maps_a = model_dict[model_a]
                maps_b = model_dict[model_b]

                # Skip and record model pairs with different coordinates or tile counts.
                coords_a = maps_a["coords"]
                coords_b = maps_b["coords"]
                if coords_a.shape != coords_b.shape or not np.array_equal(coords_a, coords_b):
                    print(f"[Skip cross-model] coords mismatch: {case_name}/{checkpoint_name} "
                          f"{model_a} vs {model_b}")
                    continue

                num_tiles = maps_a["num_tiles"]

                # ---- correlation ----
                def add_corr(rows, metric_name):
                    if metric_name in maps_a and metric_name in maps_b:
                        pear, spear = safe_corr(maps_a[metric_name], maps_b[metric_name])
                        rows.append({
                            "case_name": case_name,
                            "checkpoint_name": checkpoint_name,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric_name": metric_name,
                            "pearson": pear,
                            "spearman": spear,
                            "num_tiles": num_tiles,
                        })

                for m in DELTA_METRICS:
                    add_corr(delta_corr_rows, m)
                for m in COSINE_METRICS:
                    add_corr(cosine_corr_rows, m)
                for m in GEOMETRY_METRICS:
                    add_corr(geometry_corr_rows, m)

                # ---- top-k overlap ----
                for metric_name in TOPK_METRICS:
                    if metric_name not in maps_a or metric_name not in maps_b:
                        continue
                    for tkp in top_k_percents:
                        set_a = top_k_indices(maps_a[metric_name], tkp)
                        set_b = top_k_indices(maps_b[metric_name], tkp)
                        inter, jac, dice = jaccard_dice(set_a, set_b)
                        topk_rows.append({
                            "case_name": case_name,
                            "checkpoint_name": checkpoint_name,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric_name": metric_name,
                            "top_k_percent": tkp,
                            "overlap_count": inter,
                            "jaccard": jac,
                            "dice": dice,
                            "num_tiles": num_tiles,
                        })

    corr_fields = ["case_name", "checkpoint_name", "model_a", "model_b",
                   "metric_name", "pearson", "spearman", "num_tiles"]
    topk_fields = ["case_name", "checkpoint_name", "model_a", "model_b",
                   "metric_name", "top_k_percent", "overlap_count",
                   "jaccard", "dice", "num_tiles"]

    write_csv(out_dir / "cross_model_delta_correlation.csv", delta_corr_rows, corr_fields)
    write_csv(out_dir / "cross_model_cosine_change_correlation.csv", cosine_corr_rows, corr_fields)
    if geometry_corr_rows:
        write_csv(out_dir / "cross_model_geometry_change_correlation.csv",
                  geometry_corr_rows, corr_fields)
    write_csv(out_dir / "cross_model_topk_overlap.csv", topk_rows, topk_fields)

    print(f"  cross_model_delta_correlation.csv         : {len(delta_corr_rows)} rows")
    print(f"  cross_model_cosine_change_correlation.csv : {len(cosine_corr_rows)} rows")
    print(f"  cross_model_geometry_change_correlation.csv: {len(geometry_corr_rows)} rows")
    print(f"  cross_model_topk_overlap.csv              : {len(topk_rows)} rows")


def cross_model_cka_analysis(cross_model_store, out_dir):
    """Compare CKA before and after fine-tuning for model pairs with identical tile coordinates. Load features one case and checkpoint group at a time. A positive change indicates greater representational similarity."""
    rows = []

    for (case_name, checkpoint_name), model_dict in sorted(cross_model_store.items()):
        model_names = sorted(model_dict.keys())
        if len(model_names) < 2:
            continue

        # Read each model's feature pair once per case and checkpoint group.
        loaded = {}
        for m in model_names:
            maps = model_dict[m]
            try:
                bf, bc = sort_by_coords(*load_h5(str(maps["before_h5"])))
                af, ac = sort_by_coords(*load_h5(str(maps["after_h5"])))
                loaded[m] = {
                    "before": bf.astype(np.float32),
                    "after": af.astype(np.float32),
                    "coords": bc,
                    "feature_dim": bf.shape[1],
                }
            except Exception as e:
                print(f"[Skip cross-model CKA] load failed {case_name}/{checkpoint_name} {m}: {e}")

        loaded_names = sorted(loaded.keys())
        for i in range(len(loaded_names)):
            for j in range(i + 1, len(loaded_names)):
                ma, mb = loaded_names[i], loaded_names[j]
                da, db = loaded[ma], loaded[mb]

                # Require identical coordinates to ensure rows are aligned.
                if da["coords"].shape != db["coords"].shape or \
                        not np.array_equal(da["coords"], db["coords"]):
                    print(f"[Skip cross-model CKA] coords mismatch: "
                          f"{case_name}/{checkpoint_name} {ma} vs {mb}")
                    continue

                cka_before = linear_cka(da["before"], db["before"])
                cka_after = linear_cka(da["after"], db["after"])
                rows.append({
                    "case_name": case_name,
                    "checkpoint_name": checkpoint_name,
                    "model_a": ma,
                    "model_b": mb,
                    "cka_before": cka_before,
                    "cka_after": cka_after,
                    "cka_change": cka_after - cka_before,
                    "num_tiles": da["coords"].shape[0],
                    "feature_dim_a": da["feature_dim"],
                    "feature_dim_b": db["feature_dim"],
                })

        del loaded  # Release the features for this group.

    fields = ["case_name", "checkpoint_name", "model_a", "model_b",
              "cka_before", "cka_after", "cka_change",
              "num_tiles", "feature_dim_a", "feature_dim_b"]
    write_csv(out_dir / "cross_model_cka.csv", rows, fields)
    print(f"  cross_model_cka.csv                       : {len(rows)} rows")


# =====================================================================
# Arguments and configuration
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-model before/after fine-tuning feature space comparison."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config file listing models / dirs / outdir.")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--smooth-sigma", type=float, default=DEFAULT_SMOOTH_SIGMA)
    parser.add_argument("--darken-bg", type=float, default=DEFAULT_DARKEN_BG)
    parser.add_argument("--sample", type=int, default=DEFAULT_PCA_SAMPLE,
                        help="Total sampled tile-feature rows for model PCA fitting AND "
                             "delta/cosine color-limit estimation (memory-friendly). "
                             f"<=0 means use safe default ({DEFAULT_PCA_SAMPLE}); it does "
                             "NOT mean full read anymore.")
    parser.add_argument("--enable-geometry-change", action="store_true", default=DEFAULT_ENABLE_GEOMETRY_CHANGE,
                        help="Enable optional local geometry change map.")
    parser.add_argument("--num-landmarks", type=int, default=DEFAULT_NUM_LANDMARKS)
    parser.add_argument("--geometry-batch-size", type=int, default=DEFAULT_GEOMETRY_BATCH_SIZE,
                        help="Row batch size for geometry change (limits [batch, L] memory).")
    parser.add_argument("--top-k-percents", type=float, nargs="+", default=DEFAULT_TOP_K_PERCENTS,
                        help="top-k percentages for changed tile overlap.")
    parser.add_argument("--csv-only", action="store_true",default=DEFAULT_CSV_ONLY,
                        help="Only write CSV metrics; skip all PNG/overlay maps, "
                             "model PCA fitting and color-limit estimation to reduce memory.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Enable CSV-only mode through the CLI or configuration.
    if config.get("csv_only", False):
        args.csv_only = True

    models = config["models"]
    slide_dir = config.get("slide_dir", None)
    outdir = Path(config["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    per_model_root = outdir / "per_model"
    cross_model_dir = outdir / "cross_model_summary"
    per_model_root.mkdir(parents=True, exist_ok=True)
    cross_model_dir.mkdir(parents=True, exist_ok=True)

    if args.csv_only:
        print("[Mode] csv-only: skip all figure outputs (PNG/overlay); CSV metrics only.")

    # Skip scanning for slide images when overlays are disabled.
    slide_files = {}
    slide_dir_exists = False
    if (not args.csv_only) and slide_dir:
        slide_dir = Path(slide_dir)
        slide_dir_exists = slide_dir.exists()
        if slide_dir_exists:
            slide_files = collect_files_by_stem(
                slide_dir,
                [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"],
            )
        else:
            print(f"[Warning] slide_dir does not exist: {slide_dir}. No overlays.")
        if openslide is None:
            print("[Warning] openslide not installed. No overlays will be produced.")

    # Process each encoder.
    merged_store = {}   # (case, checkpoint) -> {model_name: cross_model_maps}
    all_cka_rows = []

    for model_cfg in models:
        cross_model_store, cka_rows = process_one_model(
            model_cfg=model_cfg,
            slide_files=slide_files,
            slide_dir_exists=slide_dir_exists,
            args=args,
            per_model_root=per_model_root,
        )
        all_cka_rows.extend(cka_rows)
        for key, mdict in cross_model_store.items():
            merged_store.setdefault(key, {}).update(mdict)

    # Compare correlations and top-k overlaps across encoders.
    print("\n" + "=" * 60)
    print("CROSS-MODEL ANALYSIS")
    print("=" * 60)
    cross_model_analysis(merged_store, cross_model_dir, args)

    # Compare CKA between encoders before and after fine-tuning.
    cross_model_cka_analysis(merged_store, cross_model_dir)

    # Summarize within-model CKA before and after fine-tuning.
    cka_fields = ["model_name", "case_name", "checkpoint_name",
                  "cka_before_after", "cka_change_score", "num_tiles", "feature_dim"]
    write_csv(cross_model_dir / "cka_before_after_summary.csv", all_cka_rows, cka_fields)
    print(f"  cka_before_after_summary.csv              : {len(all_cka_rows)} rows")

    print("\n========== Done ==========")
    print(f"Output root: {outdir}")
    if args.csv_only:
        print(f"  per_model/           : summary_metrics.csv only (figures skipped)")
    else:
        print(f"  per_model/           : per-model PCA maps + scalar maps + summary_metrics.csv")
    print(f"  cross_model_summary/ : correlation / topk overlap / within-model CKA / cross-model CKA CSVs")


if __name__ == "__main__":
    main()
