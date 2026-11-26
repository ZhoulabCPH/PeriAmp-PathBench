import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from overlay_pdf import (  # noqa: E402
    OverlayLayers,
    build_rgb_overlay_layers,
    build_scalar_overlay_layers,
    save_overlay_outputs,
)

try:
    import openslide
except ImportError:
    openslide = None


def load_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
        coords = f["coords"][:]
    return features, coords


def sort_by_coords(features: np.ndarray, coords: np.ndarray):
    """Sort by y and then x so matched before and after arrays have the same tile order."""
    idx = np.lexsort((coords[:, 0], coords[:, 1]))
    return features[idx], coords[idx]


def fit_shared_pca_and_rgb_scaler(
        before_features: np.ndarray,
        after_features: np.ndarray,
        sample: int = 0,
        random_state: int = 42,
):
    all_features = np.concatenate([before_features, after_features], axis=0)

    if sample and sample < len(all_features):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(all_features), size=sample, replace=False)
        fit_features = all_features[idx]
    else:
        fit_features = all_features

    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(fit_features)

    before_pca = pca.transform(before_features)
    after_pca = pca.transform(after_features)

    all_pca = np.concatenate([before_pca, after_pca], axis=0)
    rgb_scaler = MinMaxScaler(clip=True)
    rgb_scaler.fit(all_pca)

    return pca, rgb_scaler, before_pca, after_pca

def unique_path_list(paths):
    """Remove duplicate paths while preserving input order."""
    seen = set()
    unique = []
    for p in paths:
        key = str(Path(p).resolve())
        if key not in seen:
            seen.add(key)
            unique.append(Path(p))
    return unique


def fit_global_pca_and_rgb_scaler_from_paths(
        h5_paths,
        sample: int = 0,
        random_state: int = 42,
):
    """Fit a common PCA and RGB scaler across the compared HDF5 feature bags."""
    feature_arrays = []
    feature_dim = None

    for h5_path in h5_paths:
        features, _ = load_h5(str(h5_path))

        if features.ndim != 2:
            raise ValueError(f"features must be 2D in {h5_path}, got shape {features.shape}")

        if feature_dim is None:
            feature_dim = features.shape[1]
        elif features.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dim mismatch in global PCA: "
                f"{h5_path} has dim {features.shape[1]}, expected {feature_dim}"
            )

        feature_arrays.append(features)

    all_features = np.concatenate(feature_arrays, axis=0)

    if sample and sample < len(all_features):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(all_features), size=sample, replace=False)
        fit_features = all_features[idx]
    else:
        fit_features = all_features

    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(fit_features)

    all_pca = np.concatenate([pca.transform(x) for x in feature_arrays], axis=0)

    rgb_scaler = MinMaxScaler(clip=True)
    rgb_scaler.fit(all_pca)

    return pca, rgb_scaler


def robust_percentile_limits(values, low=1, high=99):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(values, [low, high])

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0

    if vmin == vmax:
        vmin = np.min(values)
        vmax = np.max(values)

    if vmin == vmax:
        eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps

    return float(vmin), float(vmax)


def compute_global_scalar_limits(jobs):
    """Compute shared feature-change and cosine similarity color limits for the before/after comparisons."""
    all_delta = []
    all_cosine = []

    for job in jobs:
        before_features, before_coords = sort_by_coords(*load_h5(str(job["before_h5"])))
        after_features, after_coords = sort_by_coords(*load_h5(str(job["after_h5"])))

        validate_inputs(before_features, before_coords, after_features, after_coords)

        delta = np.linalg.norm(after_features - before_features, axis=1)

        before_norm = np.linalg.norm(before_features, axis=1)
        after_norm = np.linalg.norm(after_features, axis=1)
        denom = np.maximum(before_norm * after_norm, 1e-12)
        cosine_sim = np.sum(before_features * after_features, axis=1) / denom

        all_delta.append(delta)
        all_cosine.append(cosine_sim)

    all_delta = np.concatenate(all_delta, axis=0)
    all_cosine = np.concatenate(all_cosine, axis=0)

    delta_limits = robust_percentile_limits(all_delta, low=1, high=99)
    cosine_limits = robust_percentile_limits(all_cosine, low=1, high=99)

    return delta_limits, cosine_limits

def normalize_rgb_with_shared_scaler(x: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    x = scaler.transform(x)
    return np.clip(x, 0.0, 1.0)


def make_rgb_canvas_from_coords(
        rgb: np.ndarray,
        coords: np.ndarray,
        tile_size: int = 256,
        downsample: int = 16,
        background_color=(1.0, 1.0, 1.0),
):
    coords = coords.astype(np.int64)
    min_x = coords[:, 0].min()
    min_y = coords[:, 1].min()
    max_x = coords[:, 0].max()
    max_y = coords[:, 1].max()

    tile_draw = max(1, tile_size // downsample)
    width = int((max_x - min_x) // downsample + tile_draw)
    height = int((max_y - min_y) // downsample + tile_draw)

    canvas = np.ones((height, width, 3), dtype=np.float32)
    canvas[:] = np.array(background_color, dtype=np.float32)

    mask = np.zeros((height, width), dtype=bool)

    for color, (x, y) in zip(rgb, coords):
        x0 = int((x - min_x) // downsample)
        y0 = int((y - min_y) // downsample)
        x1 = min(width, x0 + tile_draw)
        y1 = min(height, y0 + tile_draw)
        canvas[y0:y1, x0:x1, :] = color
        mask[y0:y1, x0:x1] = True

    return canvas, mask, (min_x, min_y), (width, height)


def make_scalar_canvas_from_coords(
        values: np.ndarray,
        coords: np.ndarray,
        tile_size: int = 256,
        downsample: int = 16,
):
    coords = coords.astype(np.int64)
    min_x = coords[:, 0].min()
    min_y = coords[:, 1].min()
    max_x = coords[:, 0].max()
    max_y = coords[:, 1].max()

    tile_draw = max(1, tile_size // downsample)
    width = int((max_x - min_x) // downsample + tile_draw)
    height = int((max_y - min_y) // downsample + tile_draw)

    canvas = np.full((height, width), np.nan, dtype=np.float32)

    for v, (x, y) in zip(values, coords):
        x0 = int((x - min_x) // downsample)
        y0 = int((y - min_y) // downsample)
        x1 = min(width, x0 + tile_draw)
        y1 = min(height, y0 + tile_draw)
        canvas[y0:y1, x0:x1] = v

    return canvas


def save_rgb_canvas(canvas: np.ndarray, out_path: Path):
    img = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def save_scalar_map(
        canvas: np.ndarray,
        out_path: Path,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        colorbar: bool = True,
        title: str = "",
):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="white")

    plt.figure(figsize=(10, 10))
    im = plt.imshow(canvas, cmap=cm, vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def build_rgb_overlay(
        slide_path: str,
        pca_canvas: np.ndarray,
        mask: np.ndarray,
        origin_xy,
        downsample: int,
        alpha: float = 0.45,
        smooth_sigma: float = 0.0,
        darken_bg: float = 1.0,
) -> OverlayLayers:
    return build_rgb_overlay_layers(
        slide_path=slide_path,
        rgb_canvas=pca_canvas,
        mask=mask,
        origin_xy=origin_xy,
        downsample=downsample,
        alpha=alpha,
        smooth_sigma=smooth_sigma,
        darken_bg=darken_bg,
    )


def build_scalar_overlay(
        slide_path: str,
        scalar_canvas: np.ndarray,
        mask: np.ndarray,
        origin_xy,
        downsample: int,
        cmap: str = "viridis",
        vmin=None,
        vmax=None,
        alpha: float = 0.45,
        smooth_sigma: float = 0.0,
        darken_bg: float = 1.0,
) -> OverlayLayers:
    return build_scalar_overlay_layers(
        slide_path=slide_path,
        attr_canvas=scalar_canvas,
        mask=mask,
        origin_xy=origin_xy,
        downsample=downsample,
        cmap=cmap,
        symmetric=False,
        alpha=alpha,
        smooth_sigma=smooth_sigma,
        darken_bg=darken_bg,
        clip_lo=0.0,
        clip_hi=100.0,
        vmin=vmin,
        vmax=vmax,
    )


def validate_inputs(
        before_features: np.ndarray,
        before_coords: np.ndarray,
        after_features: np.ndarray,
        after_coords: np.ndarray,
):
    if before_features.ndim != 2 or after_features.ndim != 2:
        raise ValueError("features must both be 2D arrays with shape [N, D].")

    if before_coords.ndim != 2 or before_coords.shape[1] != 2:
        raise ValueError("before coords must have shape [N, 2].")

    if after_coords.ndim != 2 or after_coords.shape[1] != 2:
        raise ValueError("after coords must have shape [N, 2].")

    if before_features.shape[1] != after_features.shape[1]:
        raise ValueError(
            f"Feature dims do not match: {before_features.shape[1]} vs {after_features.shape[1]}"
        )

    if len(before_features) != len(after_features):
        raise ValueError(
            f"Tile counts do not match: {len(before_features)} vs {len(after_features)}"
        )

    if len(before_coords) != len(after_coords):
        raise ValueError(
            f"Coord counts do not match: {len(before_coords)} vs {len(after_coords)}"
        )

    if not np.array_equal(before_coords, after_coords):
        raise ValueError(
            "Coords are not exactly the same even after sorting. Please check your data."
        )


def collect_files_by_stem(folder: Path, extensions):
    file_dict = {}
    for ext in extensions:
        for p in folder.glob(f"*{ext}"):
            file_dict[p.stem] = p
    return file_dict


def process_one_case(
        case_name: str,
        before_h5: Path,
        after_h5: Path,
        slide_path: Path,
        args,
        outdir: Path,
        fixed_pca=None,
        fixed_rgb_scaler=None,
        delta_limits=None,
        cosine_limits=None,
):
    print(f"\n========== Processing: {case_name} ==========")
    print(f"before_h5: {before_h5}")
    print(f"after_h5 : {after_h5}")
    if slide_path is not None:
        print(f"slide    : {slide_path}")
    else:
        print("slide    : None")

    before_features, before_coords = sort_by_coords(*load_h5(str(before_h5)))
    after_features, after_coords = sort_by_coords(*load_h5(str(after_h5)))

    validate_inputs(before_features, before_coords, after_features, after_coords)

    print(f"Loaded and aligned before features: {before_features.shape}")
    print(f"Loaded and aligned after features:  {after_features.shape}")

    if fixed_pca is None or fixed_rgb_scaler is None:
        pca, rgb_scaler, before_pca, after_pca = fit_shared_pca_and_rgb_scaler(
            before_features=before_features,
            after_features=after_features,
            sample=args.sample,
            random_state=42,
        )
        pca_name = "Shared PCA"
    else:
        pca = fixed_pca
        rgb_scaler = fixed_rgb_scaler
        before_pca = pca.transform(before_features)
        after_pca = pca.transform(after_features)
        pca_name = "Global PCA"

    before_rgb = normalize_rgb_with_shared_scaler(before_pca, rgb_scaler)
    after_rgb = normalize_rgb_with_shared_scaler(after_pca, rgb_scaler)

    explained = pca.explained_variance_ratio_
    print(f"{pca_name} explained variance ratio:", explained)

    # 1) Before PCA map
    before_canvas, before_mask, origin_xy, _ = make_rgb_canvas_from_coords(
        rgb=before_rgb,
        coords=before_coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
        background_color=(1.0, 1.0, 1.0),
    )
    save_rgb_canvas(before_canvas, outdir / f"{case_name}_before_pca_map.png")

    # 2) After PCA map
    after_canvas, after_mask, _, _ = make_rgb_canvas_from_coords(
        rgb=after_rgb,
        coords=after_coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
        background_color=(1.0, 1.0, 1.0),
    )
    save_rgb_canvas(after_canvas, outdir / f"{case_name}_after_pca_map.png")

    # 3) Feature change magnitude map
    delta = np.linalg.norm(after_features - before_features, axis=1)

    if delta_limits is None:
        delta_vmin, delta_vmax = robust_percentile_limits(delta, low=1, high=99)
    else:
        delta_vmin, delta_vmax = delta_limits

    delta_canvas = make_scalar_canvas_from_coords(
        values=delta,
        coords=before_coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
    )
    save_scalar_map(
        delta_canvas,
        outdir / f"{case_name}_feature_change_magnitude_map.png",
        cmap="magma",
        vmin=delta_vmin,
        vmax=delta_vmax,
        colorbar=True,
        title=f"{case_name} - Feature Change Magnitude",
    )

    # 4) Cosine similarity map
    before_norm = np.linalg.norm(before_features, axis=1)
    after_norm = np.linalg.norm(after_features, axis=1)
    denom = np.maximum(before_norm * after_norm, 1e-12)
    cosine_sim = np.sum(before_features * after_features, axis=1) / denom

    if cosine_limits is None:
        cos_vmin, cos_vmax = robust_percentile_limits(cosine_sim, low=2, high=98)
    else:
        cos_vmin, cos_vmax = cosine_limits

    cosine_canvas = make_scalar_canvas_from_coords(
        values=cosine_sim,
        coords=before_coords,
        tile_size=args.tile_size,
        downsample=args.downsample,
    )
    save_scalar_map(
        cosine_canvas,
        outdir / f"{case_name}_cosine_similarity_map.png",
        cmap="viridis",
        vmin=cos_vmin,
        vmax=cos_vmax,
        colorbar=True,
        title=f"{case_name} - Cosine Similarity",
    )

    # Optional overlays
    if slide_path is not None and openslide is not None:
        before_layers = build_rgb_overlay(
            slide_path=str(slide_path),
            pca_canvas=before_canvas,
            mask=before_mask,
            origin_xy=origin_xy,
            downsample=args.downsample,
            alpha=args.alpha,
            smooth_sigma=args.smooth_sigma,
            darken_bg=args.darken_bg
        )
        after_layers = build_rgb_overlay(
            slide_path=str(slide_path),
            pca_canvas=after_canvas,
            mask=after_mask,
            origin_xy=origin_xy,
            downsample=args.downsample,
            alpha=args.alpha,
            smooth_sigma=args.smooth_sigma,
            darken_bg=args.darken_bg
        )

        delta_layers = build_scalar_overlay(
            slide_path=str(slide_path),
            scalar_canvas=delta_canvas,
            mask=before_mask,
            origin_xy=origin_xy,
            downsample=args.downsample,
            cmap="magma",
            vmin=delta_vmin,
            vmax=delta_vmax,
            alpha=args.alpha,
            smooth_sigma=args.smooth_sigma,
            darken_bg=args.darken_bg
        )

        cosine_layers = build_scalar_overlay(
            slide_path=str(slide_path),
            scalar_canvas=cosine_canvas,
            mask=before_mask,
            origin_xy=origin_xy,
            downsample=args.downsample,
            cmap="viridis",
            vmin=cos_vmin,
            vmax=cos_vmax,
            alpha=args.alpha,
            smooth_sigma=args.smooth_sigma,
            darken_bg=args.darken_bg
        )

        Image.fromarray((before_layers.thumbnail * 255).astype(np.uint8)).save(
            outdir / f"{case_name}_wsi_thumbnail.png"
        )
        save_overlay_outputs(
            before_layers,
            outdir / f"{case_name}_before_pca_overlay.png",
            alpha=args.alpha,
        )
        save_overlay_outputs(
            after_layers,
            outdir / f"{case_name}_after_pca_overlay.png",
            alpha=args.alpha,
        )
        save_overlay_outputs(
            delta_layers,
            outdir / f"{case_name}_feature_change_magnitude_overlay.png",
            alpha=args.alpha,
        )
        save_overlay_outputs(
            cosine_layers,
            outdir / f"{case_name}_cosine_similarity_overlay.png",
            alpha=args.alpha,
        )

    # Save explained variance
    plt.figure(figsize=(6, 4))
    plt.bar([1, 2, 3], explained)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"{case_name} - {pca_name} Explained Variance")
    plt.xticks([1, 2, 3])
    plt.tight_layout()
    plt.savefig(outdir / f"{case_name}_shared_pca_explained_variance.png", dpi=300)
    plt.close()

    print(f"Done: {case_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch compare before/after features on multiple WSIs and output maps."
    )

    parser.add_argument(
        "--before-dir",
        type=str,
        default=r"data/features/gigapath",
        help="Folder containing H5 files before fine-tuning"
    )
    parser.add_argument(
        "--after-dirs",
        type=str,
        nargs="+",
        default=[
            # r"data/features/finetune_virchow2/early",
            # r"data/features/finetune_virchow2/later",
            # r"data/features/finetune_virchow2/mid",
            # r"data/features/finetune_virchow2/teacher"
            r"data/features/finetune_Gigapath/late",
            r"data/features/finetune_Gigapath/early",
            r"data/features/finetune_Gigapath/mid"
        ],
        help="One or more folders containing H5 files after fine-tuning"
    )
    parser.add_argument(
        "--slide-dir",
        type=str,
        default=r"data/slides",
        help="Folder containing WSI files"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=r"outputs/interpretability/pca/gigapath",
        help="Output directory"
    )

    parser.add_argument("--tile-size", type=int, default=256, help="Tile size")
    parser.add_argument("--downsample", type=int, default=16, help="Visualization downsample factor")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha")
    parser.add_argument("--smooth-sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--darken-bg", type=float, default=0.8, help="Background darkening factor")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional random sample size for global PCA fitting. 0 means use all.",
    )

    args = parser.parse_args()

    before_dir = Path(args.before_dir)
    after_dirs = [Path(p) for p in args.after_dirs]
    slide_dir = Path(args.slide_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not before_dir.exists():
        raise FileNotFoundError(f"before-dir does not exist: {before_dir}")

    for after_dir in after_dirs:
        if not after_dir.exists():
            raise FileNotFoundError(f"after-dir does not exist: {after_dir}")

    if not slide_dir.exists():
        print(f"Warning: slide-dir does not exist: {slide_dir}")
        print("Will continue without overlay outputs.")

    before_files = collect_files_by_stem(before_dir, [".h5"])

    if not before_files:
        raise RuntimeError(f"No h5 files found in before-dir: {before_dir}")

    slide_files = {}
    if slide_dir.exists():
        slide_files = collect_files_by_stem(
            slide_dir,
            [".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"]
        )

    processed = 0
    skipped = 0
    total_matched = 0

    used_after_tags = {}
    jobs = []

    # Collect the before and after comparison jobs.
    for after_dir in after_dirs:
        after_files = collect_files_by_stem(after_dir, [".h5"])
        common_h5_names = sorted(set(before_files.keys()) & set(after_files.keys()))

        if not common_h5_names:
            print(f"[Warning] No matched h5 filenames found for after-dir: {after_dir}")
            continue

        after_tag = after_dir.name
        if after_tag in used_after_tags:
            used_after_tags[after_tag] += 1
            after_tag = f"{after_tag}_{used_after_tags[after_tag]}"
        else:
            used_after_tags[after_tag] = 1

        after_outdir = outdir / after_tag
        after_outdir.mkdir(parents=True, exist_ok=True)

        print("\n==============================")
        print(f"After directory: {after_dir}")
        print(f"Output subdir  : {after_outdir}")
        print(f"Matched cases  : {len(common_h5_names)}")
        print("==============================")

        total_matched += len(common_h5_names)

        for case_name in common_h5_names:
            slide_path = slide_files.get(case_name, None)

            jobs.append({
                "case_name": case_name,
                "before_h5": before_files[case_name],
                "after_h5": after_files[case_name],
                "after_dir": after_dir,
                "after_outdir": after_outdir,
                "slide_path": slide_path,
            })

    if not jobs:
        raise RuntimeError("No matched before-after h5 comparisons found.")

    # Skip jobs with mismatched coordinates, tile counts, or feature widths.
    valid_jobs = []

    print("\n========== Pre-validating matched comparisons ==========")

    for job in jobs:
        case_name = job["case_name"]
        before_h5 = job["before_h5"]
        after_h5 = job["after_h5"]
        after_dir = job["after_dir"]

        try:
            before_features, before_coords = sort_by_coords(*load_h5(str(before_h5)))
            after_features, after_coords = sort_by_coords(*load_h5(str(after_h5)))
            validate_inputs(before_features, before_coords, after_features, after_coords)
            valid_jobs.append(job)
        except Exception as e:
            skipped += 1
            print(f"[Error] Invalid comparison skipped: {case_name} in {after_dir}: {e}")

    jobs = valid_jobs

    if not jobs:
        raise RuntimeError("No valid before-after h5 comparisons left after validation.")

    # Fit one PCA and RGB scaler to the features from all valid jobs.
    global_feature_paths = []

    for job in jobs:
        global_feature_paths.append(job["before_h5"])
        global_feature_paths.append(job["after_h5"])

    global_feature_paths = unique_path_list(global_feature_paths)

    print("\n========== Fitting global PCA color space ==========")
    print(f"Unique h5 files used for global PCA: {len(global_feature_paths)}")

    global_pca, global_rgb_scaler = fit_global_pca_and_rgb_scaler_from_paths(
        h5_paths=global_feature_paths,
        sample=args.sample,
        random_state=42,
    )

    print("Global PCA explained variance ratio:", global_pca.explained_variance_ratio_)

    # Estimate shared color limits across the comparisons.
    print("\n========== Computing global scalar color limits ==========")

    delta_limits, cosine_limits = compute_global_scalar_limits(jobs)

    print(f"Global feature change magnitude limits: {delta_limits}")
    print(f"Global cosine similarity limits        : {cosine_limits}")

    # Process the jobs using the shared PCA, scaler, and color limits.
    print("\n========== Processing all comparisons with global color space ==========")

    for job in jobs:
        case_name = job["case_name"]
        before_h5 = job["before_h5"]
        after_h5 = job["after_h5"]
        slide_path = job["slide_path"]
        after_dir = job["after_dir"]
        after_outdir = job["after_outdir"]

        if slide_dir.exists() and slide_path is None:
            print(
                f"[Warning] No matching WSI found for case: {case_name}. "
                "Will output maps only, no overlay."
            )

        try:
            process_one_case(
                case_name=case_name,
                before_h5=before_h5,
                after_h5=after_h5,
                slide_path=slide_path,
                args=args,
                outdir=after_outdir,
                fixed_pca=global_pca,
                fixed_rgb_scaler=global_rgb_scaler,
                delta_limits=delta_limits,
                cosine_limits=cosine_limits,
            )
            processed += 1
        except Exception as e:
            skipped += 1
            print(f"[Error] Failed processing {case_name} in {after_dir}: {e}")

    print("\n========== Summary ==========")
    print(f"Matched comparisons   : {total_matched}")
    print(f"Valid comparisons     : {len(jobs)}")
    print(f"Processed successfully: {processed}")
    print(f"Skipped / failed      : {skipped}")
    print(f"Output directory      : {outdir}")


if __name__ == "__main__":
    main()
