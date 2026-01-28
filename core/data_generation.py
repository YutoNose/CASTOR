"""
Synthetic data generation for inverse prediction experiments.

Generates data with carefully controlled Ectopic and Intrinsic anomalies
to validate the inverse prediction approach.

Uses Zero-Inflated Negative Binomial (ZINB) distribution to mimic
realistic single-cell/spatial transcriptomics count data.
"""

import warnings
import numpy as np
from typing import Tuple, Optional


def generate_synthetic_data(
    n_spots: int = 3000,
    n_genes: int = 500,
    n_ectopic: int = 100,
    n_intrinsic: int = 300,
    n_modules: int = 20,
    min_distance_factor: float = 0.5,
    random_state: int = 42,
    # ZINB parameters
    dispersion: float = 2.0,
    dropout_rate: float = 0.3,
    library_size_mean: float = 10000,
    library_size_cv: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ST data with known anomalies using ZINB distribution.

    Uses Zero-Inflated Negative Binomial (ZINB) to generate realistic
    count data with:
    - Spatial gene expression modules
    - Library size variation
    - Technical dropout (zero-inflation)
    - Overdispersion (negative binomial)

    Ectopic anomalies: Expression copied from a distant location
        -> Position prediction should fail (predicts donor location)

    Intrinsic anomalies: Aberrant expression pattern
        -> Expression is globally unusual (doesn't belong anywhere)

    Parameters
    ----------
    n_spots : int
        Number of spots
    n_genes : int
        Number of genes
    n_ectopic : int
        Number of ectopic anomalies (actual count may be less if not enough distant spots)
    n_intrinsic : int
        Number of intrinsic anomalies
    n_modules : int
        Number of spatial expression modules
    min_distance_factor : float
        Minimum distance factor for ectopic donors (fraction of grid side)
    random_state : int
        Random seed
    dispersion : float
        Negative binomial dispersion parameter (r). Lower = more overdispersion.
    dropout_rate : float
        Zero-inflation probability (technical dropout rate)
    library_size_mean : float
        Mean total counts per spot
    library_size_cv : float
        Coefficient of variation for library size

    Returns
    -------
    X : np.ndarray
        Count matrix [n_spots, n_genes] (ZINB counts, log1p normalized)
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels : np.ndarray
        0=normal, 1=ectopic, 2=intrinsic
    ectopic_idx : np.ndarray
        Indices of successfully injected ectopic anomalies
    intrinsic_idx : np.ndarray
        Indices of intrinsic anomalies
    """
    rng = np.random.RandomState(random_state)

    # Create grid coordinates with jitter
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    # Normalize coordinates to [0, 1] for consistency
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)

    # ==========================================================================
    # Step 1: Create mean expression matrix (mu) with spatial structure
    # ==========================================================================

    # Base gene expression rates (gamma distributed, mimicking real data)
    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)

    # Initialize mean matrix
    mu_matrix = np.zeros((n_spots, n_genes))
    genes_per_module = n_genes // n_modules

    for m in range(n_modules):
        # Random center for this spatial module
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)

        # Spatial decay pattern (Gaussian-like)
        spatial_weight = np.exp(-distances ** 2 / 0.1)
        spatial_weight = spatial_weight / (spatial_weight.max() + 1e-10)  # Normalize to [0, 1]

        gene_start = m * genes_per_module
        gene_end = min(gene_start + genes_per_module, n_genes)

        for g in range(gene_start, gene_end):
            # Combine base rate with spatial pattern
            base = gene_base_rates[g]
            spatial_effect = rng.uniform(0.5, 3.0)  # How much this gene responds to spatial pattern

            mu_matrix[:, g] = base * (1 + spatial_effect * spatial_weight)

    # Handle remaining genes (if n_genes not divisible by n_modules)
    remaining_start = n_modules * genes_per_module
    if remaining_start < n_genes:
        for g in range(remaining_start, n_genes):
            # Random spatial pattern for remaining genes
            center = rng.rand(2)
            distances = np.linalg.norm(coords_norm - center, axis=1)
            spatial_weight = np.exp(-distances ** 2 / rng.uniform(0.05, 0.2))
            mu_matrix[:, g] = gene_base_rates[g] * (1 + rng.uniform(0.5, 2.0) * spatial_weight)

    # ==========================================================================
    # Step 2: Apply library size variation
    # ==========================================================================

    library_sizes = rng.lognormal(
        mean=np.log(library_size_mean),
        sigma=library_size_cv,
        size=(n_spots, 1)
    )

    # Scale mu to achieve target library sizes
    current_totals = mu_matrix.sum(axis=1, keepdims=True) + 1e-8
    mu_matrix = mu_matrix * (library_sizes / current_totals)

    # ==========================================================================
    # Step 3: Initialize labels
    # ==========================================================================

    labels = np.zeros(n_spots, dtype=int)

    # ==========================================================================
    # Step 4: Inject Ectopic anomalies (copy mu from distant spot)
    # ==========================================================================

    min_dist = min_distance_factor  # Now in normalized coordinates [0, 1]
    candidate_ectopic = rng.choice(n_spots, min(n_ectopic, n_spots), replace=False)
    successful_ectopic = []

    # CRITICAL FIX: Copy mu_matrix BEFORE the loop to avoid cascading contamination
    # Without this, if spot A copies from spot B, then spot C might copy from
    # the already-modified spot A, getting B's expression instead of A's original
    mu_original = mu_matrix.copy()

    for idx in candidate_ectopic:
        # Find spots that are far away (in normalized coordinates)
        distances = np.linalg.norm(coords_norm - coords_norm[idx], axis=1)
        distant_spots = np.where(distances > min_dist)[0]

        if len(distant_spots) > 0:
            donor = rng.choice(distant_spots)
            # Copy from ORIGINAL mu_matrix to avoid cascading contamination
            mu_matrix[idx] = mu_original[donor].copy()
            labels[idx] = 1
            successful_ectopic.append(idx)

    ectopic_idx = np.array(successful_ectopic)

    if len(ectopic_idx) < n_ectopic:
        warnings.warn(
            f"Only {len(ectopic_idx)}/{n_ectopic} ectopic anomalies could be injected "
            f"(not enough distant spots with min_distance_factor={min_distance_factor})"
        )

    # ==========================================================================
    # Step 5: Inject Intrinsic anomalies (aberrant expression)
    # ==========================================================================

    remaining = np.setdiff1d(np.arange(n_spots), ectopic_idx)
    intrinsic_idx = rng.choice(remaining, min(n_intrinsic, len(remaining)), replace=False)

    for idx in intrinsic_idx:
        # Affect 15-30% of genes with aberrant expression
        n_affected = rng.randint(max(20, n_genes // 7), max(50, n_genes // 3))
        affected_genes = rng.choice(n_genes, n_affected, replace=False)

        # Strong upregulation (3-10x increase in mean)
        boost_factors = rng.uniform(3.0, 10.0, n_affected)
        mu_matrix[idx, affected_genes] *= boost_factors

        # Also downregulate some genes
        n_down = n_affected // 3
        down_genes = rng.choice(
            np.setdiff1d(np.arange(n_genes), affected_genes),
            min(n_down, n_genes - n_affected),
            replace=False
        )
        mu_matrix[idx, down_genes] *= rng.uniform(0.1, 0.3, len(down_genes))

    labels[intrinsic_idx] = 2

    # ==========================================================================
    # Step 6: Generate ZINB counts
    # ==========================================================================

    # Negative Binomial parameters
    # NB parameterization: mean = mu, variance = mu + mu^2/r
    r = dispersion
    p = r / (r + mu_matrix + 1e-12)  # Success probability

    # Generate NB counts
    X_counts = rng.negative_binomial(n=r, p=p)

    # Apply zero-inflation (dropout)
    # Dropout probability can be gene-specific and expression-dependent
    # Higher expression = lower dropout (typical in scRNA-seq)
    base_dropout = dropout_rate
    # Expression-dependent dropout: low expressed genes have higher dropout
    gene_means = mu_matrix.mean(axis=0)
    gene_dropout_rates = base_dropout * np.exp(-gene_means / (gene_means.mean() + 1e-10))
    gene_dropout_rates = np.clip(gene_dropout_rates, 0.1, 0.8)

    # Apply dropout
    dropout_mask = rng.random(X_counts.shape) < gene_dropout_rates
    X_counts[dropout_mask] = 0

    # ==========================================================================
    # Step 7: Return raw counts (log1p is applied in prepare_data)
    # ==========================================================================

    return X_counts.astype(float), coords, labels, ectopic_idx, intrinsic_idx


def generate_controlled_ectopic(
    n_spots: int = 3000,
    n_genes: int = 500,
    n_ectopic: int = 100,
    min_distance_factor: float = 0.5,
    random_state: int = 42,
    dispersion: float = 2.0,
    dropout_rate: float = 0.3,
    library_size_mean: float = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ZINB data with controlled ectopic anomalies for position prediction validation.

    Returns additional donor_positions for validation of the inverse prediction hypothesis.

    Parameters
    ----------
    n_spots : int
        Number of spots
    n_genes : int
        Number of genes
    n_ectopic : int
        Number of ectopic anomalies
    min_distance_factor : float
        Minimum distance factor for donors (in normalized coordinates)
    random_state : int
        Random seed
    dispersion : float
        NB dispersion parameter
    dropout_rate : float
        Zero-inflation rate
    library_size_mean : float
        Mean library size

    Returns
    -------
    X : np.ndarray
        Log-normalized count matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels : np.ndarray
        0=normal, 1=ectopic
    donor_positions : np.ndarray
        Donor positions for each spot [n_spots, 2]
    """
    rng = np.random.RandomState(random_state)

    # Grid coordinates
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    # Normalize coordinates
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)

    # Create spatial expression patterns (mu matrix)
    n_modules = 20
    genes_per_module = n_genes // n_modules
    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)

    mu_matrix = np.zeros((n_spots, n_genes))

    for m in range(n_modules):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-distances ** 2 / 0.1)

        gene_start = m * genes_per_module
        gene_end = min(gene_start + genes_per_module, n_genes)

        for g in range(gene_start, gene_end):
            spatial_effect = rng.uniform(0.5, 3.0)
            mu_matrix[:, g] = gene_base_rates[g] * (1 + spatial_effect * spatial_weight)

    # Library size
    library_sizes = rng.lognormal(np.log(library_size_mean), 0.3, (n_spots, 1))
    current_totals = mu_matrix.sum(axis=1, keepdims=True) + 1e-8
    mu_matrix = mu_matrix * (library_sizes / current_totals)

    # Initialize labels and donor positions
    labels = np.zeros(n_spots, dtype=int)
    donor_positions = coords.copy()

    # Inject ectopic anomalies
    min_dist = min_distance_factor
    candidate_ectopic = rng.choice(n_spots, min(n_ectopic, n_spots), replace=False)
    successful_count = 0

    # CRITICAL FIX: Copy mu_matrix BEFORE the loop to avoid cascading contamination
    mu_original = mu_matrix.copy()

    for idx in candidate_ectopic:
        distances = np.linalg.norm(coords_norm - coords_norm[idx], axis=1)
        distant = np.where(distances > min_dist)[0]

        if len(distant) > 0:
            donor = rng.choice(distant)
            # Copy from ORIGINAL mu_matrix
            mu_matrix[idx] = mu_original[donor].copy()
            donor_positions[idx] = coords[donor]
            labels[idx] = 1
            successful_count += 1

    if successful_count < n_ectopic:
        warnings.warn(
            f"Only {successful_count}/{n_ectopic} ectopic anomalies could be injected"
        )

    # Generate ZINB counts
    r = dispersion
    p = r / (r + mu_matrix + 1e-12)
    X_counts = rng.negative_binomial(n=r, p=p)

    # Dropout
    gene_means = mu_matrix.mean(axis=0)
    gene_dropout_rates = dropout_rate * np.exp(-gene_means / (gene_means.mean() + 1e-10))
    gene_dropout_rates = np.clip(gene_dropout_rates, 0.1, 0.8)
    dropout_mask = rng.random(X_counts.shape) < gene_dropout_rates
    X_counts[dropout_mask] = 0

    # Return raw counts (log1p is applied in prepare_data)
    return X_counts.astype(float), coords, labels, donor_positions


def generate_raw_counts(
    n_spots: int = 3000,
    n_genes: int = 500,
    n_ectopic: int = 100,
    n_intrinsic: int = 300,
    n_modules: int = 20,
    min_distance_factor: float = 0.5,
    random_state: int = 42,
    dispersion: float = 2.0,
    dropout_rate: float = 0.3,
    library_size_mean: float = 10000,
    library_size_cv: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ST data returning RAW counts (not log-normalized).

    Same as generate_synthetic_data but returns raw counts for custom preprocessing.

    Returns
    -------
    X_counts : np.ndarray
        Raw count matrix [n_spots, n_genes]
    X_norm : np.ndarray
        Log-normalized matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels : np.ndarray
        0=normal, 1=ectopic, 2=intrinsic
    ectopic_idx : np.ndarray
        Indices of ectopic anomalies
    intrinsic_idx : np.ndarray
        Indices of intrinsic anomalies
    """
    rng = np.random.RandomState(random_state)

    # Create grid coordinates
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)

    # Create mu matrix with spatial structure
    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.zeros((n_spots, n_genes))
    genes_per_module = n_genes // n_modules

    for m in range(n_modules):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-distances ** 2 / 0.1)

        gene_start = m * genes_per_module
        gene_end = min(gene_start + genes_per_module, n_genes)

        for g in range(gene_start, gene_end):
            spatial_effect = rng.uniform(0.5, 3.0)
            mu_matrix[:, g] = gene_base_rates[g] * (1 + spatial_effect * spatial_weight)

    # Remaining genes
    remaining_start = n_modules * genes_per_module
    for g in range(remaining_start, n_genes):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-distances ** 2 / rng.uniform(0.05, 0.2))
        mu_matrix[:, g] = gene_base_rates[g] * (1 + rng.uniform(0.5, 2.0) * spatial_weight)

    # Library size
    library_sizes = rng.lognormal(np.log(library_size_mean), library_size_cv, (n_spots, 1))
    current_totals = mu_matrix.sum(axis=1, keepdims=True) + 1e-8
    mu_matrix = mu_matrix * (library_sizes / current_totals)

    labels = np.zeros(n_spots, dtype=int)

    # Ectopic
    min_dist = min_distance_factor
    candidate_ectopic = rng.choice(n_spots, min(n_ectopic, n_spots), replace=False)
    successful_ectopic = []

    # CRITICAL FIX: Copy mu_matrix BEFORE the loop to avoid cascading contamination
    mu_original = mu_matrix.copy()

    for idx in candidate_ectopic:
        distances = np.linalg.norm(coords_norm - coords_norm[idx], axis=1)
        distant_spots = np.where(distances > min_dist)[0]
        if len(distant_spots) > 0:
            donor = rng.choice(distant_spots)
            # Copy from ORIGINAL mu_matrix
            mu_matrix[idx] = mu_original[donor].copy()
            labels[idx] = 1
            successful_ectopic.append(idx)

    ectopic_idx = np.array(successful_ectopic)

    # Intrinsic
    remaining = np.setdiff1d(np.arange(n_spots), ectopic_idx)
    intrinsic_idx = rng.choice(remaining, min(n_intrinsic, len(remaining)), replace=False)

    for idx in intrinsic_idx:
        n_affected = rng.randint(max(20, n_genes // 7), max(50, n_genes // 3))
        affected_genes = rng.choice(n_genes, n_affected, replace=False)
        mu_matrix[idx, affected_genes] *= rng.uniform(3.0, 10.0, n_affected)

        n_down = n_affected // 3
        down_genes = rng.choice(
            np.setdiff1d(np.arange(n_genes), affected_genes),
            min(n_down, n_genes - n_affected),
            replace=False
        )
        mu_matrix[idx, down_genes] *= rng.uniform(0.1, 0.3, len(down_genes))

    labels[intrinsic_idx] = 2

    # ZINB
    r = dispersion
    p = r / (r + mu_matrix + 1e-12)
    X_counts = rng.negative_binomial(n=r, p=p)

    gene_means = mu_matrix.mean(axis=0)
    gene_dropout_rates = dropout_rate * np.exp(-gene_means / (gene_means.mean() + 1e-10))
    gene_dropout_rates = np.clip(gene_dropout_rates, 0.1, 0.8)
    dropout_mask = rng.random(X_counts.shape) < gene_dropout_rates
    X_counts[dropout_mask] = 0

    X_norm = np.log1p(X_counts)

    return X_counts, X_norm, coords, labels, ectopic_idx, intrinsic_idx
