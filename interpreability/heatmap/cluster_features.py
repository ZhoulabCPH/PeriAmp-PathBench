"""
Cluster tile features for a slide cohort.
Read features and coords from HDF5. An optional case list, patch table, and tumor-region filter define the intersection of eligible tiles. The sample table accepts a patch path or a slide ID with x and y coordinates. Tiles excluded from clustering have cluster_id_full = -1.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path, PureWindowsPath
from typing import List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances_argmin,
)

from tumor_region import (
    add_tumor_region_arguments,
    apply_tumor_filter_by_case,
    rebuild_case_spans,
    tumor_filter_kwargs,
)


# ============================================================
# Default cohort clustering settings
# ============================================================

# Clustering method: kmeans, affinity_propagation, or leiden
DEFAULT_CLUSTER_METHOD = "kmeans"

# KMeans
DEFAULT_N_CLUSTERS = 0          # A positive K is fixed. Nonpositive selects K using CH and DB.
DEFAULT_AUTO_K_MIN = 5           # Minimum candidate K
DEFAULT_AUTO_K_MAX = 10          # Maximum candidate K
DEFAULT_AUTO_K_SAMPLE = 30000    # Patch sample cap for CH and DB scoring. Nonpositive uses all patches.

# Affinity Propagation
DEFAULT_AP_DAMPING = 0.8
DEFAULT_AP_PREFERENCE = -100
DEFAULT_AP_MAX_SAMPLES = 10000
DEFAULT_AP_MAX_ITER = 200
DEFAULT_AP_CONVERGENCE_ITER = 15

# Leiden
DEFAULT_LEIDEN_N_NEIGHBORS = 100
DEFAULT_LEIDEN_RESOLUTION = 0.15
DEFAULT_LEIDEN_METRIC = "euclidean"

# PCA and feature standardization
DEFAULT_PCA_DIM = 100
DEFAULT_STANDARDIZE = True

# UMAP is used for visualization only.
DEFAULT_UMAP_ENABLED = True
DEFAULT_UMAP_NEIGHBORS = 200
DEFAULT_UMAP_MIN_DIST = 0.5
DEFAULT_UMAP_METRIC = "euclidean"
DEFAULT_UMAP_SAMPLE = 60000

# Random seed
DEFAULT_SEED = 45


# ============================================================
# Default input and output paths
# ============================================================

DEFAULT_FEATURE_CSV = Path(
    r"data/run_configs/feature_gigapath.csv"
)
DEFAULT_PATCH_ROOT = Path(r"outputs/tiles/stream")
DEFAULT_OUT_DIR = Path(
    r"outputs/interpretability/tumor_cluster/double_high"
)

DEFAULT_FEATURE_LABEL_INCLUDE: List[str] = []
DEFAULT_FEATURE_LABEL_EXCLUDE: List[str] = []

# Optional case list CSV
DEFAULT_CASE_CSV = Path(
    r"data/run_configs/cases_annotated.csv"
)

# ------------------------------------------------------------
# An optional sample table can select patches by file path.
# None disables the sample table filter.
# ------------------------------------------------------------
DEFAULT_SAMPLE_TABLE: Optional[Path] = r"data/run_configs/patch_samples.xlsx"
DEFAULT_SAMPLE_SHEET = "Sheet1"
DEFAULT_SAMPLE_PATH_COLUMN = "patch_path"

# Tumor-region filtering
DEFAULT_TUMOR_FILTER = False
DEFAULT_ANNOTATION_DIR = Path(
    r"outputs/interpretability/tumor_area_output"
)
DEFAULT_ANNOTATION_CSV: Optional[Path] = None


# ============================================================
# Clustering helpers
# ============================================================

def load_feature_csv(csv_path: Path) -> dict[str, Path]:
    """Read feature directory labels and paths from a CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(f)
            if row and any(cell.strip() for cell in row)
        ]

    if not rows:
        return {}

    header = [x.lower() for x in rows[0]]
    path_keys = ("feature_path", "feature_dir", "path")
    label_keys = ("label", "feature_label", "name")

    path_idx = next((header.index(k) for k in path_keys if k in header), None)
    label_idx = next((header.index(k) for k in label_keys if k in header), None)

    if path_idx is not None and label_idx is not None:
        data_rows = rows[1:]
    else:
        path_idx, label_idx = 0, 1
        data_rows = rows

    feature_map: dict[str, Path] = {}
    for row in data_rows:
        if len(row) <= max(path_idx, label_idx):
            continue

        feature_path = row[path_idx].strip()
        label = row[label_idx].strip()

        if not feature_path or not label or feature_path.startswith("#"):
            continue

        feature_map[label] = Path(feature_path)

    return feature_map


def _normalize_higher_better(v: np.ndarray) -> np.ndarray:
    """Min-max scale a metric while preserving its order."""
    rng = float(v.max() - v.min())
    if rng <= 0:
        return np.zeros_like(v, dtype=np.float64)
    return ((v - v.min()) / rng).astype(np.float64)


def _normalize_lower_better(v: np.ndarray) -> np.ndarray:
    """Reverse a lower-is-better metric such as Davies-Bouldin before min-max scaling."""
    rng = float(v.max() - v.min())
    if rng <= 0:
        return np.zeros_like(v, dtype=np.float64)
    return ((v.max() - v) / rng).astype(np.float64)


def auto_select_k_ch_db(
    x_cluster: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    sample_for_score: int,
) -> Tuple[int, List[dict]]:
    """Fit K-means for each candidate K and select the best combined Calinski-Harabasz and Davies-Bouldin score."""
    n = x_cluster.shape[0]
    k_max_eff = max(k_min, min(k_max, max(2, n - 1)))

    if 0 < sample_for_score < n:
        rng = np.random.default_rng(seed)
        score_idx = rng.choice(n, sample_for_score, replace=False)
    else:
        score_idx = np.arange(n)

    print(
        f"[Auto-K] Testing K in [{k_min}, {k_max_eff}]  "
        f"(scoring sample: {len(score_idx)} / {n})"
    )

    raw: List[dict] = []
    for k in range(k_min, k_max_eff + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(x_cluster)
        labels_s = labels[score_idx]
        x_s = x_cluster[score_idx]
        unique = np.unique(labels_s)

        if unique.size < 2:
            ch = float("nan")
            db = float("nan")
        else:
            ch = float(calinski_harabasz_score(x_s, labels_s))
            db = float(davies_bouldin_score(x_s, labels_s))

        raw.append({
            "k": k,
            "ch": ch,
            "db": db,
            "inertia": float(km.inertia_),
        })

        print(
            f"  K={k:>3}  CH={ch:>10.2f}   DB={db:>6.4f}   "
            f"inertia={km.inertia_:>12.2f}"
        )

    ch_arr = np.array([r["ch"] for r in raw], dtype=np.float64)
    db_arr = np.array([r["db"] for r in raw], dtype=np.float64)
    ch_norm = _normalize_higher_better(ch_arr)
    db_norm = _normalize_lower_better(db_arr)
    combined = (ch_norm + db_norm) / 2.0

    for i, r in enumerate(raw):
        r["ch_norm"] = float(ch_norm[i])
        r["db_norm"] = float(db_norm[i])
        r["combined"] = float(combined[i])

    best_idx = int(np.nanargmax(combined))
    best_k = int(raw[best_idx]["k"])

    print(
        f"[Auto-K] Selected K = {best_k}  "
        f"(combined={combined[best_idx]:.4f})"
    )
    return best_k, raw


def save_k_scan_csv(out_csv: Path, records: List[dict], best_k: int) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "k", "ch", "db", "ch_norm", "db_norm",
        "combined", "inertia", "is_best",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in records:
            row = dict(r)
            row["is_best"] = 1 if r["k"] == best_k else 0
            w.writerow(row)

    print(f"[saved] {out_csv}")


def save_k_scan_plot(
    out_path_png: Path,
    records: List[dict],
    best_k: int,
    title: str,
) -> None:
    """Plot the scores for the candidate cluster counts."""
    ks = [r["k"] for r in records]
    ch_norm = [r["ch_norm"] for r in records]
    db_norm = [r["db_norm"] for r in records]
    combined = [r["combined"] for r in records]

    color_ch = "#3A82C4"
    color_db = "#F4A040"
    color_co = "#5FB04F"

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.plot(
        ks, ch_norm, marker="s", color=color_ch,
        linewidth=2, label="Calinski-Harabasz"
    )
    ax.plot(
        ks, db_norm, marker="o", color=color_db,
        linewidth=2, label="Davies-Bouldin"
    )
    ax.plot(
        ks, combined, marker="D", color=color_co,
        linewidth=2, label="Combined"
    )

    ch_peak_i = int(np.argmax(ch_norm))
    db_peak_i = int(np.argmax(db_norm))
    best_i = ks.index(best_k)
    best_y = combined[best_i]

    ax.annotate(
        f"Peak: {ch_norm[ch_peak_i]:.2f}",
        xy=(ks[ch_peak_i], ch_norm[ch_peak_i]),
        xytext=(ks[ch_peak_i] + 0.3, ch_norm[ch_peak_i] + 0.04),
        fontsize=9,
        color=color_ch,
        arrowprops=dict(
            arrowstyle="-", color=color_ch, lw=0.8
        ),
    )

    ax.annotate(
        f"Peak: {db_norm[db_peak_i]:.2f}",
        xy=(ks[db_peak_i], db_norm[db_peak_i]),
        xytext=(ks[db_peak_i] - 0.5, db_norm[db_peak_i] + 0.04),
        fontsize=9,
        color=color_db,
        arrowprops=dict(
            arrowstyle="-", color=color_db, lw=0.8
        ),
    )

    ax.axvline(
        best_k,
        color=color_co,
        linestyle="--",
        linewidth=1.2,
    )

    ax.scatter(
        [best_k],
        [best_y],
        s=280,
        facecolors="none",
        edgecolors="red",
        linewidths=2.2,
        zorder=5,
        label="_nolegend_",
    )

    ax.annotate(
        f"Optimal number of clusters: {best_k}",
        xy=(best_k, best_y),
        xytext=(best_k + 0.4, best_y - 0.18),
        fontsize=9,
        color="#3D8B3D",
        arrowprops=dict(
            arrowstyle="-",
            color=color_co,
            lw=0.8,
            alpha=0.6,
        ),
    )

    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters", fontsize=11)
    ax.set_ylabel("Normalized score", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_png.with_suffix(".pdf"))
    plt.close(fig)

    print(f"[saved] {out_path_png}")


def run_global_kmeans(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    standardize: bool,
    pca_dim: int,
    cluster_method: str = DEFAULT_CLUSTER_METHOD,
    auto_k_min: int = 0,
    auto_k_max: int = 0,
    auto_k_sample: int = 0,
    auto_k_out_dir: Optional[Path] = None,
    auto_k_prefix: Optional[str] = None,
    ap_damping: float = DEFAULT_AP_DAMPING,
    ap_preference: Optional[float] = DEFAULT_AP_PREFERENCE,
    ap_max_samples: int = DEFAULT_AP_MAX_SAMPLES,
    ap_max_iter: int = DEFAULT_AP_MAX_ITER,
    ap_convergence_iter: int = DEFAULT_AP_CONVERGENCE_ITER,
    leiden_n_neighbors: int = DEFAULT_LEIDEN_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    leiden_metric: str = DEFAULT_LEIDEN_METRIC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Cluster the selected cohort patches using K-means, affinity propagation, or Leiden."""
    cluster_method = str(cluster_method or "kmeans").lower()

    if cluster_method in {"ap", "affinity", "affinity-propagation"}:
        cluster_method = "affinity_propagation"

    if cluster_method not in {
        "kmeans",
        "affinity_propagation",
        "leiden",
    }:
        raise ValueError(
            f"Unknown clustering method: {cluster_method}，"
            f"Choose kmeans, affinity_propagation, or leiden."
        )

    x = features.astype(np.float32, copy=False)

    if standardize:
        print(
            f"[cohort] Z-score standardization, shape={x.shape}"
        )
        x = StandardScaler().fit_transform(x)

    if (
        pca_dim
        and pca_dim > 0
        and pca_dim < x.shape[1]
    ):
        print(
            f"[cohort] PCA preprocessing "
            f"{x.shape[1]} → {pca_dim}"
        )
        pca = PCA(
            n_components=pca_dim,
            random_state=seed,
        )
        x_cluster = pca.fit_transform(x)
    else:
        x_cluster = x

    info: dict = {
        "cluster_method": cluster_method,
        "auto_k_records": None,
        "best_k": None,
        "affinity_cluster_centers_indices": None,
        "affinity_fit_n_samples": None,
        "affinity_fit_sampled": False,
        "affinity_max_samples": int(ap_max_samples),
        "leiden_n_neighbors": None,
        "leiden_resolution": None,
        "leiden_metric": None,
    }

    if (
        cluster_method == "kmeans"
        and (n_clusters is None or n_clusters <= 0)
    ):
        best_k, records = auto_select_k_ch_db(
            x_cluster=x_cluster,
            k_min=int(auto_k_min),
            k_max=int(auto_k_max),
            seed=seed,
            sample_for_score=int(auto_k_sample),
        )

        info["auto_k_records"] = records
        info["best_k"] = best_k

        if (
            auto_k_out_dir is not None
            and auto_k_prefix is not None
        ):
            auto_k_out_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_k_scan_csv(
                auto_k_out_dir
                / f"{auto_k_prefix}-k_scan.csv",
                records,
                best_k,
            )

            save_k_scan_plot(
                auto_k_out_dir
                / f"{auto_k_prefix}-k_scan.png",
                records,
                best_k,
                title=(
                    f"{auto_k_prefix}  "
                    f"Calinski-Harabasz / "
                    f"Davies-Bouldin / Combined  vs.  K"
                ),
            )

        n_clusters = best_k

    if cluster_method == "kmeans":
        print(
            f"[Cohort] KMeans  "
            f"n_clusters={n_clusters}  "
            f"on shape={x_cluster.shape}"
        )

        km = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )

        cluster_id = km.fit_predict(
            x_cluster
        ).astype(np.int64)

        info["effective_n_clusters"] = int(
            n_clusters
        )

    elif cluster_method == "affinity_propagation":
        n_total = int(x_cluster.shape[0])
        ap_max_samples = int(ap_max_samples)

        if 0 < ap_max_samples < n_total:
            rng = np.random.default_rng(seed)
            fit_idx = rng.choice(
                n_total,
                ap_max_samples,
                replace=False,
            )
            fit_idx.sort()
            x_ap = x_cluster[fit_idx]

            info["affinity_fit_sampled"] = True

            print(
                f"[cohort] AffinityPropagation points: "
                f"{n_total} > "
                f"ap_max_samples={ap_max_samples}，"
                f"fitting centers on a subsample"
            )
        else:
            fit_idx = None
            x_ap = x_cluster

        info["affinity_fit_n_samples"] = int(
            x_ap.shape[0]
        )

        print(
            f"[Cohort] AffinityPropagation  "
            f"damping={ap_damping}  "
            f"preference={ap_preference}  "
            f"fit_shape={x_ap.shape}  "
            f"full_shape={x_cluster.shape}"
        )

        ap = AffinityPropagation(
            damping=float(ap_damping),
            preference=ap_preference,
            max_iter=int(ap_max_iter),
            convergence_iter=int(ap_convergence_iter),
            random_state=int(seed),
        )

        labels_fit = ap.fit_predict(
            x_ap
        ).astype(np.int64)

        centers = ap.cluster_centers_indices_

        if centers is None or centers.size == 0:
            raise RuntimeError(
                'AffinityPropagation found no usable clusters. '
                'Adjust --ap-preference '
                'or check feature scaling.'
            )

        centers = np.asarray(
            centers,
            dtype=np.int64,
        ).reshape(-1)

        if fit_idx is None:
            unique = sorted(
                int(x)
                for x in np.unique(labels_fit)
                if int(x) >= 0
            )

            if not unique:
                raise RuntimeError(
                    "AffinityPropagation "
                    'Every cluster label is invalid.'
                )

            remap = {
                old: new
                for new, old in enumerate(unique)
            }

            cluster_id = np.array(
                [
                    remap[int(x)]
                    for x in labels_fit
                ],
                dtype=np.int64,
            )

            centers_global = centers

        else:
            centers_global = fit_idx[centers]
            center_vectors = x_cluster[
                centers_global
            ]

            print(
                '[cohort] Assigning all patches to the nearest '
                f"AffinityPropagation centers "
                f"(n_centers={center_vectors.shape[0]})"
            )

            cluster_id = pairwise_distances_argmin(
                x_cluster,
                center_vectors,
            ).astype(np.int64)

        info["effective_n_clusters"] = int(
            len(centers_global)
        )

        info["affinity_cluster_centers_indices"] = [
            int(x)
            for x in np.asarray(
                centers_global
            ).reshape(-1).tolist()
        ]

        print(
            f"[cohort] AffinityPropagation found "
            f"n_clusters="
            f"{info['effective_n_clusters']}"
        )

        print(
            "[Cohort] AP cluster_centers_indices_ = "
            f"{info['affinity_cluster_centers_indices']}"
        )

    else:
        try:
            import scanpy as sc  # type: ignore
        except Exception as e:
            raise ImportError(
                'The leiden clustering method '
                'requires scanpy, anndata, '
                "igraph/leidenalg。"
            ) from e

        leiden_n_neighbors = int(
            leiden_n_neighbors
        )
        leiden_resolution = float(
            leiden_resolution
        )
        leiden_metric = str(leiden_metric)

        print(
            f"[Cohort] Leiden  "
            f"n_neighbors={leiden_n_neighbors}  "
            f"resolution={leiden_resolution}  "
            f"metric={leiden_metric}  "
            f"on shape={x_cluster.shape}"
        )

        try:
            adata = sc.AnnData(
                np.zeros(
                    (x_cluster.shape[0], 1),
                    dtype=np.float32,
                )
            )

            adata.obsm["X_cluster"] = (
                x_cluster.astype(
                    np.float32,
                    copy=False,
                )
            )

            sc.pp.neighbors(
                adata,
                n_neighbors=leiden_n_neighbors,
                use_rep="X_cluster",
                metric=leiden_metric,
                random_state=int(seed),
            )

            sc.tl.leiden(
                adata,
                resolution=leiden_resolution,
                random_state=int(seed),
                key_added="leiden",
            )

        except Exception as e:
            raise RuntimeError(
                'Leiden clustering failed. '
                'Check that the scanpy Leiden dependencies are installed '
                'and consider adjusting '
                "--leiden-n-neighbors / "
                "--leiden-resolution。"
            ) from e

        labels = adata.obs["leiden"]

        if hasattr(labels, "cat"):
            cluster_id = labels.cat.codes.to_numpy(
                dtype=np.int64
            )
        else:
            raw = labels.astype(str).to_numpy()
            unique = {
                v: i
                for i, v in enumerate(
                    sorted(set(raw))
                )
            }

            cluster_id = np.array(
                [unique[v] for v in raw],
                dtype=np.int64,
            )

        n_clusters_found = int(
            np.unique(cluster_id).size
        )

        info["effective_n_clusters"] = (
            n_clusters_found
        )
        info["leiden_n_neighbors"] = (
            leiden_n_neighbors
        )
        info["leiden_resolution"] = (
            leiden_resolution
        )
        info["leiden_metric"] = leiden_metric

        print(
            f"[cohort] Leiden found "
            f"n_clusters={n_clusters_found}"
        )

    if x_cluster.shape[1] >= 2:
        pca2 = PCA(
            n_components=2,
            random_state=seed,
        )
        x2 = pca2.fit_transform(x_cluster)
    else:
        x2 = np.column_stack([
            x_cluster[:, 0],
            np.zeros(x_cluster.shape[0]),
        ])

    return cluster_id, x2, x_cluster, info


def compute_umap_embedding(
    x: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    sample_size: int,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Run UMAP on standardized, PCA-transformed features."""
    try:
        import umap  # type: ignore
    except Exception as e:
        print(
            f"[UMAP] Unable to load umap-learn "
            f". Skipping UMAP: {e}"
        )
        print("       pip install umap-learn")
        return None, None

    n = x.shape[0]

    if 0 < sample_size < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(
            n,
            sample_size,
            replace=False,
        )
        idx.sort()
        x_in = x[idx]

        print(
            f"[UMAP] Points: {n} > "
            f"sample={sample_size}，"
            f"running UMAP on a subsample"
        )
    else:
        idx = None
        x_in = x

    print(
        f"[UMAP] Computing embedding "
        f"n_neighbors={n_neighbors}  "
        f"min_dist={min_dist}  "
        f"metric={metric}  "
        f"on shape={x_in.shape}"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(seed),
        verbose=False,
    )

    embedding = reducer.fit_transform(x_in)

    return (
        embedding.astype(
            np.float32,
            copy=False,
        ),
        idx,
    )


# ============================================================
# Patch sample table helpers
# ============================================================

_PATCH_FILENAME_RE = re.compile(
    r"^(?P<x>\d+)x_(?P<y>\d+)y(?:\.[^.]+)?$",
    re.IGNORECASE,
)


def _normalize_case_id(value) -> str:
    """Normalize equivalent numeric and string IDs to the same string, for example 1000001.0 to "1000001"."""
    if value is None:
        return ""

    s = str(value).strip()

    if not s or s.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        return ""

    if s.endswith(".0"):
        body = s[:-2]

        if body.lstrip("+-").isdigit():
            s = body

    return s


def parse_patch_path(
    patch_path,
) -> Tuple[str, int, int]:
    """Extract the slide ID and coordinates from a patch path. For example, outputs/tiles/stream/slide_001/01511x_20661y.png yields ("slide_001", 1511, 20661)."""
    if patch_path is None:
        raise ValueError('patch_path is empty.')

    s = str(patch_path).strip()

    if not s or s.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        raise ValueError('patch_path is empty.')

    # Normalize spreadsheet paths on both Windows and Linux.
    # Use Windows path rules for backslash-separated values.
    p = PureWindowsPath(
        s.replace("/", "\\")
    )

    slide_id = _normalize_case_id(
        p.parent.name
    )

    match = _PATCH_FILENAME_RE.match(
        p.name
    )

    if match is None:
        raise ValueError(
            f"Could not parse tile coordinates from filename: {p.name}；"
            f"Expected a name like 01511x_20661y.png."
        )

    x = int(match.group("x"))
    y = int(match.group("y"))

    if not slide_id:
        raise ValueError(
            f"Could not infer slide ID from path: "
            f"{patch_path}"
        )

    return slide_id, x, y


def load_sample_patch_table(
    table_path: Path,
    sheet_name: str = "Sheet1",
    path_column: str = "patch_path",
) -> Tuple[
    set[Tuple[str, int, int]],
    dict,
]:
    """Read patch keys from an Excel or CSV table. Prefer patch_path, then use the slide ID and coordinate columns."""
    table_path = Path(table_path)

    if not table_path.is_file():
        raise FileNotFoundError(
            f"Sample table not found: "
            f"{table_path}"
        )

    try:
        import pandas as pd
    except Exception as e:
        raise ImportError(
            '--sample-table requires '
            "pandas/openpyxl：\n"
            "pip install pandas openpyxl"
        ) from e

    suffix = table_path.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xlsm",
        ".xls",
    }:
        df = pd.read_excel(
            table_path,
            sheet_name=sheet_name,
            dtype=object,
        )

    elif suffix == ".csv":
        df = pd.read_csv(
            table_path,
            dtype=object,
        )

    else:
        raise ValueError(
            'Supported sample table formats are '
            ".xlsx / .xlsm / .xls / .csv，"
            f"Found: {table_path}"
        )

    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    slide_col = None
    for candidate in (
        "slide_id",
        "slideid",
        "case_id",
        "caseid",
        "patient_id",
    ):
        if candidate in columns:
            slide_col = columns[candidate]
            break

    x_col = None
    for candidate in (
        "coord_x",
        "x",
    ):
        if candidate in columns:
            x_col = columns[candidate]
            break

    y_col = None
    for candidate in (
        "coord_y",
        "y",
    ):
        if candidate in columns:
            y_col = columns[candidate]
            break

    real_path_column = None

    if path_column in df.columns:
        real_path_column = path_column

    elif path_column.lower() in columns:
        real_path_column = columns[
            path_column.lower()
        ]

    if (
        real_path_column is None
        and not (
            slide_col is not None
            and x_col is not None
            and y_col is not None
        )
    ):
        raise KeyError(
            f"Sample table has no '{path_column}'，"
            f" and no "
            f"slide ID and x/y coordinate columns.\n"
            f"Available columns: {list(df.columns)}"
        )

    keys: List[
        Tuple[str, int, int]
    ] = []

    invalid_rows: List[dict] = []

    for row_idx, row in df.iterrows():
        key = None
        error_message = None

        # ----------------------------------------------------
        # Prefer patch_path when present.
        # ----------------------------------------------------
        if real_path_column is not None:
            value = row.get(
                real_path_column
            )

            if (
                value is not None
                and str(value).strip().lower()
                not in {
                    "",
                    "nan",
                    "none",
                    "<na>",
                }
            ):
                try:
                    key = parse_patch_path(
                        value
                    )
                except Exception as e:
                    error_message = str(e)

        # ----------------------------------------------------
        # 2. fallback:
        #    slide_id + coord_x/y
        # ----------------------------------------------------
        if (
            key is None
            and slide_col is not None
            and x_col is not None
            and y_col is not None
        ):
            try:
                slide_id = _normalize_case_id(
                    row.get(slide_col)
                )

                x_raw = row.get(x_col)
                y_raw = row.get(y_col)

                if (
                    x_raw is None
                    or y_raw is None
                ):
                    raise ValueError(
                        'Missing x or y coordinate.'
                    )

                x = int(float(x_raw))
                y = int(float(y_raw))

                if slide_id:
                    key = (
                        slide_id,
                        x,
                        y,
                    )

            except Exception as e:
                if error_message is None:
                    error_message = str(e)

        if key is None:
            invalid_rows.append({
                # DataFrame index zero corresponds to spreadsheet row two.
                "excel_row": int(row_idx) + 2,
                "error": (
                    error_message
                    or 'Could not parse sample.'
                ),
            })
            continue

        keys.append(key)

    sample_keys = set(keys)

    if not sample_keys:
        raise RuntimeError(
            'No valid patch keys were found in the sample table: '
            f"{table_path}"
        )

    info = {
        "enabled": True,
        "table_path": str(table_path),
        "sheet_name": str(sheet_name),
        "path_column": str(path_column),
        "n_rows": int(len(df)),
        "n_valid_rows": int(len(keys)),
        "n_unique_patches": int(
            len(sample_keys)
        ),
        "n_duplicate_rows": int(
            len(keys) - len(sample_keys)
        ),
        "n_invalid_rows": int(
            len(invalid_rows)
        ),
        "n_cases": int(
            len({
                k[0]
                for k in sample_keys
            })
        ),
    }

    print(
        f"[Sample table] {table_path}\n"
        f"[Sample table] "
        f"rows={info['n_rows']}  "
        f"valid={info['n_valid_rows']}  "
        f"unique={info['n_unique_patches']}  "
        f"duplicate={info['n_duplicate_rows']}  "
        f"invalid={info['n_invalid_rows']}  "
        f"cases={info['n_cases']}"
    )

    if invalid_rows:
        print(
            f"[Sample table] "
            f"Could not parse {len(invalid_rows)} rows. "
            f"First ten examples:"
        )

        for item in invalid_rows[:10]:
            print(
                f"  row="
                f"{item['excel_row']}  "
                f"{item['error']}"
            )

    return sample_keys, info


def build_sample_mask(
    coords: np.ndarray,
    case_ids: np.ndarray,
    sample_keys: set[
        Tuple[str, int, int]
    ],
) -> np.ndarray:
    """Select cohort tiles by their (slide_id, x, y) keys."""
    coords = np.asarray(coords)
    case_ids = np.asarray(
        case_ids,
        dtype=object,
    ).reshape(-1)

    if (
        coords.shape[0]
        != case_ids.shape[0]
    ):
        raise ValueError(
            'coords and case_ids have different row counts: '
            f"{coords.shape[0]} vs "
            f"{case_ids.shape[0]}"
        )

    mask = np.fromiter(
        (
            (
                _normalize_case_id(
                    case_ids[i]
                ),
                int(coords[i, 0]),
                int(coords[i, 1]),
            )
            in sample_keys
            for i in range(
                coords.shape[0]
            )
        ),
        dtype=bool,
        count=coords.shape[0],
    )

    return mask


def apply_sample_mask_to_cohort(
    features: np.ndarray,
    coords: np.ndarray,
    case_ids: np.ndarray,
    per_case_info: list[dict],
    sample_mask: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
    dict,
]:
    """Apply the sample mask to the current cohort, intersecting it with any previously applied tumor filter."""
    sample_mask = np.asarray(
        sample_mask,
        dtype=bool,
    ).reshape(-1)

    n_before = int(
        coords.shape[0]
    )

    if (
        sample_mask.shape[0]
        != n_before
    ):
        raise RuntimeError(
            f"Unexpected sample_mask length: "
            f"{sample_mask.shape[0]} "
            f"!= {n_before}"
        )

    features_kept = np.asarray(
        features[sample_mask],
        dtype=np.float32,
    )

    coords_kept = np.asarray(
        coords[sample_mask],
        dtype=np.int64,
    )

    case_ids_kept = np.asarray(
        case_ids[sample_mask],
        dtype=object,
    ).reshape(-1)

    n_after = int(
        coords_kept.shape[0]
    )

    if n_after == 0:
        raise RuntimeError(
            'No patches remain for clustering '
            'after applying the sample table.\n'
            'Check the following:\n'
            '1. slide_id matches the HDF5 filename stem.\n'
            '2. x and y match the HDF5 coordinates.\n'
            '3. case_csv includes the requested cases.\n'
            '4. When tumor filtering is enabled, '
            'the selected patches lie within the annotation.'
        )

    old_by_case = {
        str(c["case"]): c
        for c in per_case_info
    }

    spans = rebuild_case_spans(
        case_ids_kept
    )

    for span in spans:
        case = str(
            span["case"]
        )

        old = old_by_case.get(
            case,
            {},
        )

        span["h5"] = str(
            old.get("h5", "")
        )

        span[
            "n_patches_before_sample_filter"
        ] = int(
            old.get(
                "n_patches",
                0,
            )
        )

        span[
            "n_patches_before_tumor_filter"
        ] = int(
            old.get(
                "n_patches_before_tumor_filter",
                old.get(
                    "n_patches",
                    0,
                ),
            )
        )

    info = {
        "enabled": True,
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": int(
            n_before - n_after
        ),
        "n_cases_before": int(
            len(per_case_info)
        ),
        "n_cases_after": int(
            len(spans)
        ),
    }

    print(
        f"[sample filter] Kept "
        f"{n_after}/{n_before} patches  "
        f"(cases={len(spans)})"
    )

    return (
        features_kept,
        coords_kept,
        case_ids_kept,
        spans,
        info,
    )


def load_case_csv(
    csv_path: Path,
) -> list[str]:
    """Read one case ID per row from the CSV."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Case CSV not found: {csv_path}"
        )

    case_ids: list[str] = []

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.reader(f)

        first_data_checked = False

        for row in reader:
            if not row:
                continue

            value = str(
                row[0]
            ).strip()

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

    case_ids = list(
        dict.fromkeys(case_ids)
    )

    if not case_ids:
        raise RuntimeError(
            'Case CSV contains no case IDs: '
            f"{csv_path}"
        )

    return case_ids


def collect_h5_files(
    feature_dir: Path,
) -> list[Path]:
    if not feature_dir.is_dir():
        raise NotADirectoryError(
            'Not a valid feature directory: '
            f"{feature_dir}"
        )

    return sorted(
        feature_dir.glob("*.h5")
    )


def load_feature_cohort(
    feature_dir: Path,
    feature_label: str,
    selected_cases: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
]:
    """Load all patch features and coordinates for an encoder from HDF5 bags."""
    all_h5_files = collect_h5_files(
        feature_dir
    )

    h5_map = {
        p.stem: p
        for p in all_h5_files
    }

    h5_files = [
        h5_map[case_id]
        for case_id in selected_cases
        if case_id in h5_map
    ]

    missing_cases = [
        case_id
        for case_id in selected_cases
        if case_id not in h5_map
    ]

    if missing_cases:
        print(
            f"[warning] feature_label="
            f"{feature_label} has "
            f"{len(missing_cases)} cases "
            f"without matching HDF5 files: "
        )

        for case_id in missing_cases:
            print(
                f"  - {case_id}.h5"
            )

    if not h5_files:
        raise RuntimeError(
            f"feature_label={feature_label}: "
            f"none of the cases from case_csv "
            f"has a matching HDF5 file: {feature_dir}"
        )

    print(
        f"[case filter] "
        f"feature_label={feature_label}: "
        f"requested {len(selected_cases)} cases, "
        f"found {len(h5_files)} cases"
    )

    features_list = []
    coords_list = []
    case_ids_list = []
    per_case_info = []

    feature_dim = None
    row_cursor = 0

    for h5_path in h5_files:
        with h5py.File(
            h5_path,
            "r",
        ) as f:
            if (
                "features" not in f
                or "coords" not in f
            ):
                print(
                    f"[skip] {h5_path.name} "
                    f" is missing features or coords."
                )
                continue

            features = np.asarray(
                f["features"][:]
            )

            coords = np.asarray(
                f["coords"][:]
            )

        if (
            features.ndim != 2
            or coords.ndim != 2
            or coords.shape[1] < 2
        ):
            print(
                f"[skip] {h5_path.name} "
                f" has invalid feature or coordinate shapes: "
                f"{features.shape}, {coords.shape}"
            )
            continue

        if (
            features.shape[0]
            != coords.shape[0]
            or features.shape[0] == 0
        ):
            print(
                f"[skip] {h5_path.name} "
                f" has empty or mismatched feature and coordinate arrays."
            )
            continue

        if feature_dim is None:
            feature_dim = int(
                features.shape[1]
            )

        elif (
            features.shape[1]
            != feature_dim
        ):
            print(
                f"[skip] {h5_path.name} "
                f"feat_dim="
                f"{features.shape[1]} "
                f"!= {feature_dim}"
            )
            continue

        n = int(
            features.shape[0]
        )

        features_list.append(
            features.astype(
                np.float32,
                copy=False,
            )
        )

        coords_list.append(
            coords[:, :2].astype(
                np.int64,
                copy=False,
            )
        )

        case_ids_list.append(
            np.full(
                n,
                h5_path.stem,
                dtype=object,
            )
        )

        per_case_info.append({
            "case": h5_path.stem,
            "h5": str(h5_path),
            "n_patches": n,
            "start_row": row_cursor,
            "end_row": row_cursor + n,
        })

        row_cursor += n

        print(
            f"  - {h5_path.stem:<12} "
            f"n_patches={n:>5} "
            f"feat_dim={features.shape[1]}"
        )

    if not features_list:
        raise RuntimeError(
            f"feature_label={feature_label} "
            f" contains no usable HDF5 cases."
        )

    return (
        np.concatenate(
            features_list,
            axis=0,
        ),
        np.concatenate(
            coords_list,
            axis=0,
        ),
        np.concatenate(
            case_ids_list,
            axis=0,
        ),
        per_case_info,
    )


def filter_cohort_to_tumor_region(
    features: np.ndarray,
    coords: np.ndarray,
    case_ids: np.ndarray,
    per_case_info: list[dict],
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
    dict,
    np.ndarray,
]:
    """Filter tiles with per-case tumor annotation CSVs and rebuild case spans. Return a mask aligned to the original cohort, with all entries true when filtering is disabled."""
    kwargs = tumor_filter_kwargs(
        args
    )

    n_before = int(
        np.asarray(coords).shape[0]
    )

    if not kwargs["enabled"]:
        print(
            '[tumor] Filtering disabled. '
            'Using all patches.'
        )

        tumor_mask = np.ones(
            n_before,
            dtype=bool,
        )

        return (
            features,
            coords,
            case_ids,
            per_case_info,
            {
                "enabled": False,
                "n_before": n_before,
                "n_after": n_before,
                "n_cases": int(
                    len(per_case_info)
                ),
                "n_cases_empty": 0,
            },
            tumor_mask,
        )

    annotation_dir = kwargs.get(
        "annotation_dir"
    )

    annotation_csv = kwargs.get(
        "annotation_csv"
    )

    print(
        f"[tumor] Enabled "
        f"annotation_dir={annotation_dir}  "
        f"annotation_csv="
        f"{annotation_csv or '(lookup by case)'}"
    )

    (
        coords_kept,
        arrays_kept,
        tumor_mask,
        info,
    ) = apply_tumor_filter_by_case(
        coords,
        case_ids,
        arrays={
            "features": features,
            "case_ids": case_ids,
        },
        per_case_info=per_case_info,
        **kwargs,
    )

    tumor_mask = np.asarray(
        tumor_mask,
        dtype=bool,
    ).reshape(-1)

    if (
        tumor_mask.shape[0]
        != n_before
    ):
        raise RuntimeError(
            f"Unexpected tumor_mask length: "
            f"{tumor_mask.shape[0]} "
            f" versus original patch count "
            f"{n_before}"
        )

    features_kept = np.asarray(
        arrays_kept["features"],
        dtype=np.float32,
    )

    case_ids_kept = np.asarray(
        arrays_kept["case_ids"],
        dtype=object,
    ).reshape(-1)

    h5_by_case = {
        str(c["case"]): str(
            c.get("h5", "")
        )
        for c in per_case_info
    }

    n_before_by_case = {
        str(c["case"]): int(
            c["n_patches"]
        )
        for c in per_case_info
    }

    spans = rebuild_case_spans(
        case_ids_kept
    )

    for span in spans:
        case = str(
            span["case"]
        )

        span["h5"] = h5_by_case.get(
            case,
            "",
        )

        span[
            "n_patches_before_tumor_filter"
        ] = n_before_by_case.get(
            case,
            0,
        )

    print(
        f"[tumor] Total retained: "
        f"{info['n_after']}/"
        f"{info['n_before']} patches  "
        f"(cases={info['n_cases']}, "
        f"empty annotations="
        f"{info['n_cases_empty']})"
    )

    kept_cases = {
        str(s["case"])
        for s in spans
    }

    dropped = [
        str(c["case"])
        for c in per_case_info
        if str(c["case"])
        not in kept_cases
    ]

    if dropped:
        print(
            f"[tumor] "
            f"cases with no patches after filtering: "
            f"({len(dropped)}): "
            + ", ".join(dropped)
        )

    return (
        features_kept,
        coords_kept,
        case_ids_kept,
        spans,
        info,
        tumor_mask,
    )


def _jsonable_dict(
    info: Optional[dict],
) -> Optional[dict]:
    if not info:
        return None

    out = {}

    for key, value in info.items():
        if isinstance(
            value,
            (bool, np.bool_),
        ):
            out[str(key)] = bool(value)

        elif isinstance(
            value,
            (int, np.integer),
        ):
            out[str(key)] = int(value)

        elif isinstance(
            value,
            Path,
        ):
            out[str(key)] = str(value)

        else:
            out[str(key)] = value

    return out


def save_result(
    out_dir: Path,
    cohort_prefix: str,
    feature_label: str,
    cluster_id: np.ndarray,
    coords: np.ndarray,
    case_ids: np.ndarray,
    per_case_info: list,
    cluster_info: dict,
    args: argparse.Namespace,
    source_feature_dir: Path,
    pca_embedding: np.ndarray,
    umap_embedding=None,
    umap_indices=None,
    tumor_filter_info: Optional[dict] = None,
    coords_full: Optional[np.ndarray] = None,
    case_ids_full: Optional[np.ndarray] = None,
    cluster_id_full: Optional[np.ndarray] = None,
    tumor_mask: Optional[np.ndarray] = None,
    per_case_info_full: Optional[list] = None,
    sample_filter_info: Optional[dict] = None,
    sample_mask: Optional[np.ndarray] = None,
    tumor_region_mask: Optional[np.ndarray] = None,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz_path = (
        out_dir
        / f"{cohort_prefix}-cluster_result.npz"
    )

    json_path = (
        out_dir
        / f"{cohort_prefix}-cluster_result.json"
    )

    arrays = {
        "cluster_id": np.asarray(
            cluster_id,
            dtype=np.int64,
        ),
        "coords": np.asarray(
            coords,
            dtype=np.int64,
        ),
        "case_ids": np.asarray(
            case_ids,
            dtype=object,
        ),
        "pca_embedding": np.asarray(
            pca_embedding,
            dtype=np.float32,
        ),
    }

    # --------------------------------------------------------
    # Preserve full-slide coordinates in the dump.
    # --------------------------------------------------------
    if coords_full is not None:
        arrays["coords_full"] = np.asarray(
            coords_full,
            dtype=np.int64,
        )

    if case_ids_full is not None:
        arrays["case_ids_full"] = np.asarray(
            case_ids_full,
            dtype=object,
        )

    if cluster_id_full is not None:
        arrays["cluster_id_full"] = np.asarray(
            cluster_id_full,
            dtype=np.int64,
        )

    # tumor_mask marks tiles included in clustering.
    if tumor_mask is not None:
        arrays["tumor_mask"] = np.asarray(
            tumor_mask,
            dtype=bool,
        )

    # --------------------------------------------------------
    # Include the sample mask when a sample table is used.
    # --------------------------------------------------------
    if sample_mask is not None:
        arrays["sample_mask"] = np.asarray(
            sample_mask,
            dtype=bool,
        )

    if tumor_region_mask is not None:
        arrays[
            "tumor_region_mask"
        ] = np.asarray(
            tumor_region_mask,
            dtype=bool,
        )

    if umap_embedding is not None:
        arrays["umap_embedding"] = np.asarray(
            umap_embedding,
            dtype=np.float32,
        )

        arrays["umap_indices"] = (
            np.asarray(
                umap_indices,
                dtype=np.int64,
            )
            if umap_indices is not None
            else np.empty(
                0,
                dtype=np.int64,
            )
        )

    np.savez_compressed(
        npz_path,
        **arrays,
    )

    meta = {
        "schema_version": 1,
        "feature_label": feature_label,
        "cohort_prefix": cohort_prefix,
        "source_feature_dir": str(
            source_feature_dir
        ),
        "cluster_method": str(
            cluster_info.get(
                "cluster_method",
                "kmeans",
            )
        ),
        "effective_n_clusters": int(
            cluster_info.get(
                "effective_n_clusters",
                0,
            )
        ),
        "best_k": int(
            cluster_info.get(
                "best_k"
            )
            or 0
        ),
        "cluster_info": cluster_info,
        "patch_root": str(
            args.patch_root
        ),
        "case_csv": str(
            args.case_csv
        ),
        "selected_case_count": int(
            len(per_case_info)
        ),
        "tumor_filter": bool(
            getattr(
                args,
                "tumor_filter",
                DEFAULT_TUMOR_FILTER,
            )
        ),
        "annotation_dir": str(
            getattr(
                args,
                "annotation_dir",
                "",
            )
            or ""
        ),
        "annotation_csv": str(
            getattr(
                args,
                "annotation_csv",
                "",
            )
            or ""
        ),
        "tumor_filter_info": _jsonable_dict(
            tumor_filter_info
        ),

        # Record the sample table metadata.
        "sample_table": str(
            getattr(
                args,
                "sample_table",
                "",
            )
            or ""
        ),
        "sample_sheet": str(
            getattr(
                args,
                "sample_sheet",
                "",
            )
            or ""
        ),
        "sample_path_column": str(
            getattr(
                args,
                "sample_path_column",
                "",
            )
            or ""
        ),
        "sample_filter_info": _jsonable_dict(
            sample_filter_info
        ),

        "full_wsi": {
            "saved": bool(
                coords_full is not None
                and case_ids_full is not None
                and cluster_id_full is not None
                and tumor_mask is not None
            ),
            "n_patches_full": (
                int(
                    np.asarray(
                        coords_full
                    ).shape[0]
                )
                if coords_full is not None
                else 0
            ),
            "n_clustered_patches": (
                int(
                    np.sum(
                        np.asarray(
                            cluster_id_full
                        )
                        >= 0
                    )
                )
                if cluster_id_full is not None
                else int(
                    np.asarray(
                        cluster_id
                    ).shape[0]
                )
            ),
            "n_non_tumor_patches": (
                int(
                    np.sum(
                        np.asarray(
                            cluster_id_full
                        )
                        < 0
                    )
                )
                if cluster_id_full is not None
                else 0
            ),
            "non_tumor_cluster_id": -1,
        },

        "args": {
            "cluster_method": args.cluster_method,
            "n_clusters": int(
                args.n_clusters
            ),
            "auto_k_min": int(
                args.auto_k_min
            ),
            "auto_k_max": int(
                args.auto_k_max
            ),
            "auto_k_sample": int(
                args.auto_k_sample
            ),
            "pca_dim": int(
                args.pca_dim
            ),
            "standardize": bool(
                args.standardize
            ),
            "seed": int(
                args.seed
            ),
            "umap_enabled": bool(
                args.umap_enabled
            ),
            "umap_neighbors": int(
                args.umap_neighbors
            ),
            "umap_min_dist": float(
                args.umap_min_dist
            ),
            "umap_metric": str(
                args.umap_metric
            ),
            "umap_sample": int(
                args.umap_sample
            ),
            "tumor_filter": bool(
                getattr(
                    args,
                    "tumor_filter",
                    DEFAULT_TUMOR_FILTER,
                )
            ),
            "annotation_dir": str(
                getattr(
                    args,
                    "annotation_dir",
                    "",
                )
                or ""
            ),
            "annotation_csv": str(
                getattr(
                    args,
                    "annotation_csv",
                    "",
                )
                or ""
            ),
            "sample_table": str(
                getattr(
                    args,
                    "sample_table",
                    "",
                )
                or ""
            ),
            "sample_sheet": str(
                getattr(
                    args,
                    "sample_sheet",
                    "",
                )
                or ""
            ),
            "sample_path_column": str(
                getattr(
                    args,
                    "sample_path_column",
                    "",
                )
                or ""
            ),
        },

        "per_case_info": [
            {
                "case": str(
                    c["case"]
                ),
                "h5": str(
                    c.get("h5", "")
                ),
                "n_patches": int(
                    c["n_patches"]
                ),
                "n_patches_before_tumor_filter": int(
                    c.get(
                        "n_patches_before_tumor_filter",
                        c["n_patches"],
                    )
                ),
                "n_patches_before_sample_filter": int(
                    c.get(
                        "n_patches_before_sample_filter",
                        c["n_patches"],
                    )
                ),
                "start_row": int(
                    c["start_row"]
                ),
                "end_row": int(
                    c["end_row"]
                ),
            }
            for c in per_case_info
        ],

        "per_case_info_full": [
            {
                "case": str(
                    c["case"]
                ),
                "h5": str(
                    c.get("h5", "")
                ),
                "n_patches": int(
                    c["n_patches"]
                ),
                "start_row": int(
                    c["start_row"]
                ),
                "end_row": int(
                    c["end_row"]
                ),
            }
            for c in (
                per_case_info_full
                or []
            )
        ],
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[saved] {npz_path}")
    print(f"[saved] {json_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Cluster features across slides in a cohort.'
    )

    p.add_argument(
        "--feature-csv",
        type=Path,
        default=DEFAULT_FEATURE_CSV,
    )

    p.add_argument(
        "--patch-root",
        type=Path,
        default=DEFAULT_PATCH_ROOT,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    p.add_argument(
        "--feature-label-include",
        nargs="*",
        default=DEFAULT_FEATURE_LABEL_INCLUDE,
    )

    p.add_argument(
        "--feature-label-exclude",
        nargs="*",
        default=DEFAULT_FEATURE_LABEL_EXCLUDE,
    )

    p.add_argument(
        "--case-csv",
        type=Path,
        default=DEFAULT_CASE_CSV,
        help='Case list CSV with one case ID per row',
    )

    # --------------------------------------------------------
    # Optional patch sample table
    # --------------------------------------------------------
    p.add_argument(
        "--sample-table",
        type=Path,
        default=DEFAULT_SAMPLE_TABLE,
        help=(
            'Optional Excel or CSV patch list. '
            'When supplied, cluster only listed patches. '
            'Omit to use every otherwise eligible patch.'
        ),
    )

    p.add_argument(
        "--sample-sheet",
        type=str,
        default=DEFAULT_SAMPLE_SHEET,
        help=(
            'Worksheet to read from an Excel sample table'
        ),
    )

    p.add_argument(
        "--sample-path-column",
        type=str,
        default=DEFAULT_SAMPLE_PATH_COLUMN,
        help=(
            'Sample table column containing patch paths'
        ),
    )

    p.add_argument(
        "--cluster-method",
        choices=[
            "kmeans",
            "affinity_propagation",
            "ap",
            "leiden",
        ],
        default=str(
            DEFAULT_CLUSTER_METHOD
        ).lower(),
    )

    p.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
    )

    p.add_argument(
        "--auto-k-min",
        type=int,
        default=DEFAULT_AUTO_K_MIN,
    )

    p.add_argument(
        "--auto-k-max",
        type=int,
        default=DEFAULT_AUTO_K_MAX,
    )

    p.add_argument(
        "--auto-k-sample",
        type=int,
        default=DEFAULT_AUTO_K_SAMPLE,
    )

    p.add_argument(
        "--ap-damping",
        type=float,
        default=DEFAULT_AP_DAMPING,
    )

    p.add_argument(
        "--ap-preference",
        type=float,
        default=DEFAULT_AP_PREFERENCE,
    )

    p.add_argument(
        "--ap-max-samples",
        type=int,
        default=DEFAULT_AP_MAX_SAMPLES,
    )

    p.add_argument(
        "--ap-max-iter",
        type=int,
        default=DEFAULT_AP_MAX_ITER,
    )

    p.add_argument(
        "--ap-convergence-iter",
        type=int,
        default=DEFAULT_AP_CONVERGENCE_ITER,
    )

    p.add_argument(
        "--leiden-n-neighbors",
        type=int,
        default=DEFAULT_LEIDEN_N_NEIGHBORS,
    )

    p.add_argument(
        "--leiden-resolution",
        type=float,
        default=DEFAULT_LEIDEN_RESOLUTION,
    )

    p.add_argument(
        "--leiden-metric",
        type=str,
        default=DEFAULT_LEIDEN_METRIC,
    )

    p.add_argument(
        "--pca-dim",
        type=int,
        default=DEFAULT_PCA_DIM,
    )

    p.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        default=DEFAULT_STANDARDIZE,
    )

    p.add_argument(
        "--no-umap",
        dest="umap_enabled",
        action="store_false",
        default=DEFAULT_UMAP_ENABLED,
    )

    p.add_argument(
        "--umap-neighbors",
        type=int,
        default=DEFAULT_UMAP_NEIGHBORS,
    )

    p.add_argument(
        "--umap-min-dist",
        type=float,
        default=DEFAULT_UMAP_MIN_DIST,
    )

    p.add_argument(
        "--umap-metric",
        type=str,
        default=DEFAULT_UMAP_METRIC,
    )

    p.add_argument(
        "--umap-sample",
        type=int,
        default=DEFAULT_UMAP_SAMPLE,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    add_tumor_region_arguments(
        p,
        single_csv=True,
        annotation_dir_default=DEFAULT_ANNOTATION_DIR,
        annotation_csv_default=DEFAULT_ANNOTATION_CSV,
        tumor_filter_default=DEFAULT_TUMOR_FILTER,
    )

    p.set_defaults(
        tumor_filter=DEFAULT_TUMOR_FILTER
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ========================================================
    # Read the case list.
    # ========================================================
    selected_cases = load_case_csv(
        args.case_csv
    )

    print(
        f"[Case CSV] {args.case_csv}"
    )

    print(
        f"[case CSV] Read "
        f"{len(selected_cases)} case IDs."
    )

    # ========================================================
    # Load the sample table once when supplied.
    # ========================================================
    sample_keys = None
    sample_table_info = None

    if (
        getattr(
            args,
            "sample_table",
            None,
        )
        is not None
    ):
        (
            sample_keys,
            sample_table_info,
        ) = load_sample_patch_table(
            table_path=args.sample_table,
            sheet_name=args.sample_sheet,
            path_column=args.sample_path_column,
        )

        sample_cases = {
            case
            for case, _, _ in sample_keys
        }

        selected_case_set = set(
            selected_cases
        )

        not_in_case_csv = sorted(
            sample_cases
            - selected_case_set
        )

        if not_in_case_csv:
            print(
                f"[Sample table] "
                f"{len(not_in_case_csv)} slides "
                f"are absent from case_csv and will be skipped."
            )

            print(
                "[Sample table] "
                'First 20 slides absent from case_csv:'
            )

            for case in not_in_case_csv[:20]:
                print(
                    f"  - {case}"
                )

    feature_map = load_feature_csv(
        args.feature_csv
    )

    if not feature_map:
        raise RuntimeError(
            'No entries found in feature_csv: '
            f"{args.feature_csv}"
        )

    feature_labels = sorted(
        feature_map
    )

    if args.feature_label_include:
        feature_labels = [
            x
            for x in feature_labels
            if x in args.feature_label_include
        ]

    if args.feature_label_exclude:
        feature_labels = [
            x
            for x in feature_labels
            if x not in args.feature_label_exclude
        ]

    if not feature_labels:
        raise RuntimeError(
            'No feature labels remain after filtering.'
        )

    tumor_on = bool(
        getattr(
            args,
            "tumor_filter",
            DEFAULT_TUMOR_FILTER,
        )
    )

    print(
        f"[config] Tumor filtering: "
        f"{'enabled at ' + str(args.annotation_dir) if tumor_on else 'disabled'}"
    )

    if sample_keys is None:
        print(
            '[config] Sample table disabled. '
            'Using all patches that pass the other filters.'
        )
    else:
        print(
            f"[config] Sample table enabled at "
            f"{args.sample_table}"
        )

    for feature_label in feature_labels:
        feature_dir = Path(
            feature_map[feature_label]
        )

        print(
            f"\n[Cohort] "
            f"feature_label={feature_label} "
            f"dir={feature_dir}"
        )

        (
            features,
            coords,
            case_ids,
            per_case_info,
        ) = load_feature_cohort(
            feature_dir,
            feature_label,
            selected_cases,
        )

        # --------------------------------------------------------
        # Full-slide spatial metadata
        # --------------------------------------------------------
        coords_full = np.asarray(
            coords,
            dtype=np.int64,
        ).copy()

        case_ids_full = np.asarray(
            case_ids,
            dtype=object,
        ).copy()

        per_case_info_full = [
            dict(c)
            for c in per_case_info
        ]

        # ========================================================
        # Build the sample mask on the full slide.
        #
        # Defer feature filtering until after tumor filtering.
        # The tumor filter receives the full feature arrays.
        # ========================================================
        sample_mask_full = None
        sample_filter_info = None

        if sample_keys is not None:
            sample_mask_full = build_sample_mask(
                coords=coords_full,
                case_ids=case_ids_full,
                sample_keys=sample_keys,
            )

            n_sample_matched_full = int(
                sample_mask_full.sum()
            )

            matched_keys = {
                (
                    _normalize_case_id(
                        case_ids_full[i]
                    ),
                    int(
                        coords_full[i, 0]
                    ),
                    int(
                        coords_full[i, 1]
                    ),
                )
                for i in np.flatnonzero(
                    sample_mask_full
                )
            }

            missing_keys = (
                sample_keys
                - matched_keys
            )

            sample_filter_info = dict(
                sample_table_info
                or {}
            )

            sample_filter_info.update({
                "n_loaded_h5_patches": int(
                    coords_full.shape[0]
                ),
                "n_matched_before_tumor_filter": (
                    n_sample_matched_full
                ),
                "n_unique_sample_keys_matched": int(
                    len(matched_keys)
                ),
                "n_unique_sample_keys_not_found": int(
                    len(missing_keys)
                ),
            })

            print(
                f"[Sample table ↔ H5] "
                f"Matched "
                f"{len(matched_keys)}/"
                f"{len(sample_keys)} "
                f" unique patches."
            )

            if missing_keys:
                print(
                    f"[Sample table ↔ H5] "
                    f"There are {len(missing_keys)} patches "
                    f"without matches in the HDF5 cohort."
                )

                print(
                    "[Sample table ↔ H5] "
                    'First 20 unmatched patches:'
                )

                for (
                    case,
                    x,
                    y,
                ) in sorted(
                    missing_keys
                )[:20]:
                    print(
                        f"  {case}  "
                        f"x={x}  y={y}"
                    )

            if (
                n_sample_matched_full
                == 0
            ):
                raise RuntimeError(
                    'No sample-table patches '
                    'matched the current HDF5 cohort.\n'
                    'Check case_csv, HDF5 filenames, and coordinates.'
                )

        # ========================================================
        # Apply the tumor filter.
        # ========================================================
        (
            features,
            coords,
            case_ids,
            per_case_info,
            tumor_filter_info,
            tumor_region_mask,
        ) = filter_cohort_to_tumor_region(
            features,
            coords,
            case_ids,
            per_case_info,
            args,
        )

        tumor_region_mask = np.asarray(
            tumor_region_mask,
            dtype=bool,
        ).reshape(-1)

        # ========================================================
        # Apply the sample table selection.
        #
        # Cluster patches in the intersection of all enabled filters.
        #   case_csv
        # Include the tumor region when tumor filtering is enabled.
        # Include the patch list when a sample table is supplied.
        #
        # Skip the sample-table filter when no table is provided.
        # ========================================================
        if sample_mask_full is not None:
            sample_mask_after_tumor = (
                sample_mask_full[
                    tumor_region_mask
                ]
            )

            (
                features,
                coords,
                case_ids,
                per_case_info,
                sample_stage_info,
            ) = apply_sample_mask_to_cohort(
                features=features,
                coords=coords,
                case_ids=case_ids,
                per_case_info=per_case_info,
                sample_mask=sample_mask_after_tumor,
            )

            sample_filter_info.update({
                "n_before_sample_filter_after_tumor": int(
                    sample_stage_info[
                        "n_before"
                    ]
                ),
                "n_after_sample_filter": int(
                    sample_stage_info[
                        "n_after"
                    ]
                ),
                "n_removed_by_sample_filter": int(
                    sample_stage_info[
                        "n_removed"
                    ]
                ),
                "n_cases_after_sample_filter": int(
                    sample_stage_info[
                        "n_cases_after"
                    ]
                ),
            })

            # Track the final clustered subset in full-slide coordinates.
            cluster_mask_full = (
                tumor_region_mask
                & sample_mask_full
            )

        else:
            # ----------------------------------------------------
            # When the sample table is absent, use the tumor-filtered set.
            # Preserve the same full-slide indexing.
            # ----------------------------------------------------
            cluster_mask_full = (
                tumor_region_mask
            )

        cohort_prefix = (
            f"cohort-{feature_label}"
        )

        out_dir = (
            args.out_dir
            / f"cohort_{feature_label}"
        )

        cluster_id, pca_embedding, x_cluster, cluster_info = (
            run_global_kmeans(
                features=features,
                n_clusters=args.n_clusters,
                seed=args.seed,
                standardize=args.standardize,
                pca_dim=args.pca_dim,
                cluster_method=args.cluster_method,
                auto_k_min=args.auto_k_min,
                auto_k_max=args.auto_k_max,
                auto_k_sample=args.auto_k_sample,
                auto_k_out_dir=out_dir,
                auto_k_prefix=cohort_prefix,
                ap_damping=args.ap_damping,
                ap_preference=args.ap_preference,
                ap_max_samples=args.ap_max_samples,
                ap_max_iter=args.ap_max_iter,
                ap_convergence_iter=args.ap_convergence_iter,
                leiden_n_neighbors=args.leiden_n_neighbors,
                leiden_resolution=args.leiden_resolution,
                leiden_metric=args.leiden_metric,
            )
        )

        # ========================================================
        # Map cluster labels back to the full slide.
        # ========================================================
        cluster_mask_full = np.asarray(
            cluster_mask_full,
            dtype=bool,
        ).reshape(-1)

        if (
            cluster_mask_full.shape[0]
            != coords_full.shape[0]
        ):
            raise RuntimeError(
                f"cluster_mask_full length "
                f"{cluster_mask_full.shape[0]} "
                f" differs from full-slide patch count "
                f"{coords_full.shape[0]}"
            )

        if (
            int(
                cluster_mask_full.sum()
            )
            != cluster_id.shape[0]
        ):
            raise RuntimeError(
                f"cluster_mask_full selected count "
                f"{int(cluster_mask_full.sum())} "
                f" differs from clustered patch count "
                f"{cluster_id.shape[0]}"
            )

        cluster_id_full = np.full(
            coords_full.shape[0],
            -1,
            dtype=np.int64,
        )

        cluster_id_full[
            cluster_mask_full
        ] = cluster_id

        print(
            f"[Full WSI dump] "
            f"total={len(cluster_id_full)}  "
            f"clustered="
            f"{int(cluster_mask_full.sum())}  "
            f"outside/unselected="
            f"{int((~cluster_mask_full).sum())}"
        )

        # ========================================================
        # Generate the UMAP embedding.
        # ========================================================
        umap_embedding = None
        umap_indices = None

        if args.umap_enabled:
            (
                umap_embedding,
                umap_indices,
            ) = compute_umap_embedding(
                x=x_cluster,
                seed=args.seed,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                metric=args.umap_metric,
                sample_size=args.umap_sample,
            )

        # ========================================================
        # Write clustering outputs.
        # ========================================================
        save_result(
            out_dir=out_dir,
            cohort_prefix=cohort_prefix,
            feature_label=feature_label,
            cluster_id=cluster_id,
            coords=coords,
            case_ids=case_ids,
            per_case_info=per_case_info,
            cluster_info=cluster_info,
            args=args,
            source_feature_dir=feature_dir,
            pca_embedding=pca_embedding,
            umap_embedding=umap_embedding,
            umap_indices=umap_indices,
            tumor_filter_info=tumor_filter_info,
            coords_full=coords_full,
            case_ids_full=case_ids_full,
            cluster_id_full=cluster_id_full,

            # Store the legacy field for downstream consumers.
            # tumor_mask continues to identify the subset that was clustered.
            # The mask is aligned to the full slide.
            tumor_mask=cluster_mask_full,

            per_case_info_full=per_case_info_full,

            # Record sample-table outputs when enabled.
            sample_filter_info=sample_filter_info,

            sample_mask=(
                sample_mask_full
                if sample_mask_full is not None
                else None
            ),

            tumor_region_mask=(
                tumor_region_mask
                if sample_mask_full is not None
                else None
            ),
        )


if __name__ == "__main__":
    main()
