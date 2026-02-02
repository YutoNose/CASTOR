"""
Real spatial transcriptomics data loading and processing.

Supports:
- 10x Genomics Visium public datasets
- spatialLIBD DLPFC data
- Slide-seq data

For semi-synthetic validation: inject artificial ectopic anomalies into real data.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import urllib.request
import gzip
import shutil

# Data cache directory
CACHE_DIR = Path(__file__).parent.parent / "data_cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_file(url: str, filename: str, force: bool = False) -> Path:
    """Download a file to cache directory."""
    cache_dir = ensure_cache_dir()
    filepath = cache_dir / filename

    if filepath.exists() and not force:
        print(f"Using cached: {filename}")
        return filepath

    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to: {filepath}")
    return filepath


# =============================================================================
# 10x Genomics Visium Sample Data (via scanpy)
# =============================================================================

VISIUM_DATASETS = {
    "human_lymph_node": {
        "sample_id": "V1_Human_Lymph_Node",
        "description": "Human Lymph Node (10x Genomics)",
    },
    "mouse_brain_coronal": {
        "sample_id": "V1_Adult_Mouse_Brain",
        "description": "Adult Mouse Brain Coronal (10x Genomics)",
    },
    "mouse_brain_sagittal_posterior": {
        "sample_id": "V1_Mouse_Brain_Sagittal_Posterior",
        "description": "Mouse Brain Sagittal Posterior (10x Genomics)",
    },
    "mouse_brain_sagittal_anterior": {
        "sample_id": "V1_Mouse_Brain_Sagittal_Anterior",
        "description": "Mouse Brain Sagittal Anterior (10x Genomics)",
    },
    "human_breast_cancer": {
        "sample_id": "V1_Breast_Cancer_Block_A_Section_1",
        "description": "Human Breast Cancer (10x Genomics)",
    },
}


def download_visium_dataset(dataset_name: str, force: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Download and load a 10x Visium public dataset via scanpy.

    Parameters
    ----------
    dataset_name : str
        One of: 'human_lymph_node', 'mouse_brain_coronal', 'mouse_brain_sagittal_posterior',
                'mouse_brain_sagittal_anterior', 'human_breast_cancer'
    force : bool
        Force re-download (not used with scanpy, included for API compatibility)

    Returns
    -------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    gene_names : List[str]
        Gene names
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required. Install with: pip install scanpy")

    if dataset_name not in VISIUM_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(VISIUM_DATASETS.keys())}")

    info = VISIUM_DATASETS[dataset_name]
    sample_id = info["sample_id"]

    print(f"Loading {dataset_name} via scanpy...")

    # Load via scanpy's built-in function
    adata = sc.datasets.visium_sge(sample_id=sample_id)

    # Extract expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    # Extract spatial coordinates
    coords = adata.obsm['spatial']

    # Extract gene names
    gene_names = list(adata.var_names)

    print(f"Loaded {dataset_name}: {X.shape[0]} spots, {X.shape[1]} genes")
    return X, coords, gene_names


# =============================================================================
# spatialLIBD DLPFC Data (Alternative: download from Bioconductor)
# =============================================================================

def load_dlpfc_sample(sample_id: str = "151673") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load spatialLIBD DLPFC sample data.

    This requires the spatialLIBD data to be downloaded first.
    Use R to download: spatialLIBD::fetch_data(type = "spe")

    For now, we provide a function to load from CSV exports.

    Parameters
    ----------
    sample_id : str
        Sample ID (e.g., "151673")

    Returns
    -------
    X : np.ndarray
        Expression matrix
    coords : np.ndarray
        Spatial coordinates
    layer_labels : np.ndarray
        Layer annotations (1-6, WM)
    gene_names : List[str]
        Gene names
    """
    cache_dir = ensure_cache_dir()
    data_dir = cache_dir / "dlpfc"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"DLPFC data not found at {data_dir}. "
            "Please download using R:\n"
            "  library(spatialLIBD)\n"
            "  spe <- fetch_data(type = 'spe')\n"
            "Then export the data to CSV."
        )

    # Load from exported CSVs
    expr_file = data_dir / f"{sample_id}_expression.csv"
    coord_file = data_dir / f"{sample_id}_coordinates.csv"
    label_file = data_dir / f"{sample_id}_layers.csv"

    X = pd.read_csv(expr_file, index_col=0).values
    coords = pd.read_csv(coord_file, index_col=0).values
    layer_labels = pd.read_csv(label_file, index_col=0).values.flatten()
    gene_names = list(pd.read_csv(expr_file, index_col=0).columns)

    return X, coords, layer_labels, gene_names


# =============================================================================
# Semi-Synthetic Ectopic Injection
# =============================================================================

def inject_ectopic_by_region(
    X: np.ndarray,
    coords: np.ndarray,
    region_labels: np.ndarray,
    n_ectopic: int = 100,
    source_region: Optional[int] = None,
    target_region: Optional[int] = None,
    noise_level: float = 0.0,
    mix_alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Inject ectopic anomalies by swapping expression between regions.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    region_labels : np.ndarray
        Region/cluster labels for each spot
    n_ectopic : int
        Number of ectopic spots to create
    source_region : int, optional
        Region to take expression FROM. If None, random.
    target_region : int, optional
        Region to inject expression INTO. If None, random (different from source).
    noise_level : float
        Gaussian noise to add (fraction of std)
    mix_alpha : float
        Mixing ratio (1.0 = full replacement, 0.5 = 50% mix)
    random_state : int
        Random seed

    Returns
    -------
    X_injected : np.ndarray
        Expression matrix with ectopic injections
    labels : np.ndarray
        0=normal, 1=ectopic
    donor_positions : np.ndarray
        Position where expression came from
    metadata : Dict
        Additional information about the injection
    """
    rng = np.random.RandomState(random_state)

    X_injected = X.copy().astype(float)
    n_spots = len(X)
    labels = np.zeros(n_spots, dtype=int)
    donor_positions = coords.copy()

    unique_regions = np.unique(region_labels[~pd.isna(region_labels) if isinstance(region_labels[0], float) else region_labels != ''])
    unique_regions = unique_regions[unique_regions != '']  # Remove empty

    if len(unique_regions) < 2:
        raise ValueError("Need at least 2 regions for ectopic injection")

    # Select source and target regions
    if source_region is None:
        source_region = rng.choice(unique_regions)
    if target_region is None:
        other_regions = [r for r in unique_regions if r != source_region]
        target_region = rng.choice(other_regions)

    source_idx = np.where(region_labels == source_region)[0]
    target_idx = np.where(region_labels == target_region)[0]

    if len(source_idx) == 0 or len(target_idx) == 0:
        raise ValueError(f"Source or target region has no spots")

    # Select spots to inject
    n_inject = min(n_ectopic, len(target_idx))
    inject_idx = rng.choice(target_idx, n_inject, replace=False)

    # Inject ectopic expression
    for idx in inject_idx:
        # Select random donor from source region
        donor = rng.choice(source_idx)
        donor_expr = X[donor].copy().astype(float)

        # Apply mixing
        if mix_alpha < 1.0:
            original_expr = X_injected[idx].copy()
            donor_expr = mix_alpha * donor_expr + (1 - mix_alpha) * original_expr

        # Add noise
        if noise_level > 0:
            gene_stds = np.std(X, axis=0) + 1e-8
            noise = rng.normal(0, noise_level * gene_stds)
            donor_expr = donor_expr + noise
            donor_expr = np.clip(donor_expr, 0, None)

        X_injected[idx] = donor_expr
        labels[idx] = 1
        donor_positions[idx] = coords[donor]

    metadata = {
        "source_region": source_region,
        "target_region": target_region,
        "n_injected": n_inject,
        "noise_level": noise_level,
        "mix_alpha": mix_alpha,
        "inject_indices": inject_idx,
        "source_indices": source_idx,
        "target_indices": target_idx,
    }

    print(f"Injected {n_inject} ectopic spots from region '{source_region}' to region '{target_region}'")
    return X_injected, labels, donor_positions, metadata


def inject_ectopic_by_distance(
    X: np.ndarray,
    coords: np.ndarray,
    n_ectopic: int = 100,
    min_distance_fraction: float = 0.5,
    noise_level: float = 0.0,
    mix_alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Inject ectopic anomalies by copying expression from distant spots.

    This is similar to synthetic data generation but applied to real data.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    n_ectopic : int
        Number of ectopic spots to create
    min_distance_fraction : float
        Minimum distance as fraction of coordinate range
    noise_level : float
        Gaussian noise to add (fraction of std)
    mix_alpha : float
        Mixing ratio
    random_state : int
        Random seed

    Returns
    -------
    X_injected : np.ndarray
        Expression matrix with ectopic injections
    labels : np.ndarray
        0=normal, 1=ectopic
    donor_positions : np.ndarray
        Position where expression came from
    metadata : Dict
        Additional information
    """
    rng = np.random.RandomState(random_state)

    X_injected = X.copy().astype(float)
    n_spots = len(X)
    labels = np.zeros(n_spots, dtype=int)
    donor_positions = coords.copy()

    # Calculate minimum distance threshold
    coord_range = coords.max(axis=0) - coords.min(axis=0)
    min_dist = np.mean(coord_range) * min_distance_fraction

    # Select candidate spots
    candidates = rng.choice(n_spots, min(n_ectopic * 2, n_spots), replace=False)

    injected = 0
    inject_indices = []

    for idx in candidates:
        if injected >= n_ectopic:
            break

        # Find distant spots
        distances = np.linalg.norm(coords - coords[idx], axis=1)
        distant_spots = np.where(distances > min_dist)[0]

        if len(distant_spots) == 0:
            continue

        # Select donor
        donor = rng.choice(distant_spots)
        donor_expr = X[donor].copy().astype(float)

        # Apply mixing
        if mix_alpha < 1.0:
            original_expr = X_injected[idx].copy()
            donor_expr = mix_alpha * donor_expr + (1 - mix_alpha) * original_expr

        # Add noise
        if noise_level > 0:
            gene_stds = np.std(X, axis=0) + 1e-8
            noise = rng.normal(0, noise_level * gene_stds)
            donor_expr = donor_expr + noise
            donor_expr = np.clip(donor_expr, 0, None)

        X_injected[idx] = donor_expr
        labels[idx] = 1
        donor_positions[idx] = coords[donor]
        injected += 1
        inject_indices.append(idx)

    metadata = {
        "n_injected": injected,
        "min_distance": min_dist,
        "noise_level": noise_level,
        "mix_alpha": mix_alpha,
        "inject_indices": np.array(inject_indices),
    }

    if injected < n_ectopic:
        warnings.warn(f"Only injected {injected}/{n_ectopic} ectopic spots (not enough distant spots)")

    print(f"Injected {injected} ectopic spots (min_distance={min_dist:.1f})")
    return X_injected, labels, donor_positions, metadata


# =============================================================================
# Cluster-based region annotation (when labels not available)
# =============================================================================

def cluster_spots(
    X: np.ndarray,
    coords: np.ndarray,
    n_clusters: int = 7,
    use_spatial: bool = True,
    spatial_weight: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster spots using expression and optionally spatial information.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    coords : np.ndarray
        Spatial coordinates
    n_clusters : int
        Number of clusters
    use_spatial : bool
        Whether to incorporate spatial information
    spatial_weight : float
        Weight for spatial coordinates (0-1)
    random_state : int
        Random seed

    Returns
    -------
    labels : np.ndarray
        Cluster labels
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Log-normalize expression
    X_log = np.log1p(X)

    # PCA for dimensionality reduction
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X_log)

    # Normalize features
    scaler_expr = StandardScaler()
    X_norm = scaler_expr.fit_transform(X_pca)

    if use_spatial:
        scaler_coord = StandardScaler()
        coords_norm = scaler_coord.fit_transform(coords)

        # Combine expression and spatial
        features = np.hstack([
            X_norm * (1 - spatial_weight),
            coords_norm * spatial_weight
        ])
    else:
        features = X_norm

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    return labels


# =============================================================================
# Utility functions
# =============================================================================

def filter_genes(
    X: np.ndarray,
    gene_names: List[str],
    min_cells: int = 10,
    min_counts: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """Filter genes by minimum expression."""
    gene_counts = (X > min_counts).sum(axis=0)
    keep_genes = gene_counts >= min_cells

    X_filtered = X[:, keep_genes]
    gene_names_filtered = [g for g, k in zip(gene_names, keep_genes) if k]

    print(f"Filtered genes: {len(gene_names)} -> {len(gene_names_filtered)}")
    return X_filtered, gene_names_filtered


def filter_spots(
    X: np.ndarray,
    coords: np.ndarray,
    min_genes: int = 200,
    min_counts: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter spots by minimum expression."""
    spot_genes = (X > 0).sum(axis=1)
    spot_counts = X.sum(axis=1)

    keep_spots = (spot_genes >= min_genes) & (spot_counts >= min_counts)
    keep_idx = np.where(keep_spots)[0]

    X_filtered = X[keep_spots]
    coords_filtered = coords[keep_spots]

    print(f"Filtered spots: {len(X)} -> {len(X_filtered)}")
    return X_filtered, coords_filtered, keep_idx


def subsample_genes(
    X: np.ndarray,
    gene_names: List[str],
    n_genes: int = 2000,
    method: str = "highly_variable",
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Subsample genes for computational efficiency.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix
    gene_names : List[str]
        Gene names
    n_genes : int
        Number of genes to keep
    method : str
        'highly_variable' or 'random'
    random_state : int
        Random seed

    Returns
    -------
    X_sub : np.ndarray
        Subsampled expression matrix
    gene_names_sub : List[str]
        Subsampled gene names
    """
    if X.shape[1] <= n_genes:
        return X, gene_names

    if method == "highly_variable":
        # Calculate variance-to-mean ratio (Fano factor) as simple HVG proxy
        means = X.mean(axis=0) + 1e-8
        variances = X.var(axis=0)
        fano = variances / means

        # Select top n_genes by Fano factor
        top_idx = np.argsort(fano)[-n_genes:]
        top_idx = np.sort(top_idx)  # Maintain order
    else:
        rng = np.random.RandomState(random_state)
        top_idx = np.sort(rng.choice(X.shape[1], n_genes, replace=False))

    X_sub = X[:, top_idx]
    gene_names_sub = [gene_names[i] for i in top_idx]

    print(f"Subsampled genes: {len(gene_names)} -> {len(gene_names_sub)}")
    return X_sub, gene_names_sub
