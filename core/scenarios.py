"""
Multiple synthetic data scenarios for robust validation.

Scenarios vary in:
1. Ectopic injection method (exact copy, noisy, partial, gradient)
2. Intrinsic effect size (easy, medium, hard)
3. Spatial structure (modules, cell types, continuous gradients)

All scenarios use ZINB count generation for realistic data.

This allows comprehensive validation of the inverse prediction approach.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


class EctopicType(Enum):
    """Types of ectopic anomaly injection."""
    EXACT_COPY = "exact_copy"           # mu[i] = mu[donor]
    NOISY_COPY = "noisy_copy"           # mu[i] = mu[donor] * (1 + noise)
    PARTIAL_MIX = "partial_mix"         # mu[i] = α*mu[donor] + (1-α)*mu[i]
    GRADIENT_MIX = "gradient_mix"       # Mix ratio depends on distance
    MARKER_SWAP = "marker_swap"         # Swap cell-type specific marker mu


class IntrinsicType(Enum):
    """Types of intrinsic anomaly injection."""
    LARGE_EFFECT = "large_effect"       # 3-10x boost on mu (easy)
    MEDIUM_EFFECT = "medium_effect"     # 2-5x boost on mu (medium)
    SMALL_EFFECT = "small_effect"       # 1.5-3x boost on mu (hard)
    STRESS_MODULE = "stress_module"     # Activate specific gene module


class SpatialStructure(Enum):
    """Types of spatial structure."""
    MODULES = "modules"                 # Radial modules
    CELL_TYPES = "cell_types"           # Discrete cell type regions
    GRADIENTS = "gradients"             # Continuous spatial gradients
    LAYERED = "layered"                 # Layered structure (like cortex)


@dataclass
class ScenarioConfig:
    """Configuration for a synthetic data scenario."""
    name: str
    ectopic_type: EctopicType
    intrinsic_type: IntrinsicType
    spatial_structure: SpatialStructure

    # Ectopic parameters
    ectopic_noise_level: float = 0.1      # For noisy_copy (multiplicative noise on mu)
    ectopic_mix_alpha: float = 0.7        # For partial_mix (fraction from donor)

    # Intrinsic parameters
    intrinsic_boost_range: Tuple[float, float] = (3.0, 10.0)  # Multiplicative boost on mu
    intrinsic_gene_fraction: Tuple[float, float] = (0.15, 0.30)  # Fraction of affected genes

    # ZINB parameters
    dispersion: float = 2.0
    dropout_rate: float = 0.3
    library_size_mean: float = 10000
    library_size_cv: float = 0.3

    # Spatial parameters
    n_modules: int = 20
    n_cell_types: int = 7


# Predefined scenarios for comprehensive testing
SCENARIOS = {
    "baseline": ScenarioConfig(
        name="Baseline",
        ectopic_type=EctopicType.EXACT_COPY,
        intrinsic_type=IntrinsicType.LARGE_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
    ),

    "noisy_ectopic": ScenarioConfig(
        name="Noisy Ectopic",
        ectopic_type=EctopicType.NOISY_COPY,
        intrinsic_type=IntrinsicType.LARGE_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        ectopic_noise_level=0.2,
    ),

    "partial_ectopic": ScenarioConfig(
        name="Partial Ectopic (70% donor)",
        ectopic_type=EctopicType.PARTIAL_MIX,
        intrinsic_type=IntrinsicType.LARGE_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        ectopic_mix_alpha=0.7,
    ),

    "hard_ectopic": ScenarioConfig(
        name="Hard Ectopic (50% donor + noise)",
        ectopic_type=EctopicType.PARTIAL_MIX,
        intrinsic_type=IntrinsicType.LARGE_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        ectopic_mix_alpha=0.5,
        ectopic_noise_level=0.1,
    ),

    "medium_intrinsic": ScenarioConfig(
        name="Medium Intrinsic Effect",
        ectopic_type=EctopicType.EXACT_COPY,
        intrinsic_type=IntrinsicType.MEDIUM_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        intrinsic_boost_range=(2.0, 5.0),
    ),

    "hard_intrinsic": ScenarioConfig(
        name="Hard Intrinsic Effect",
        ectopic_type=EctopicType.EXACT_COPY,
        intrinsic_type=IntrinsicType.SMALL_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        intrinsic_boost_range=(1.5, 3.0),
    ),

    "realistic_counts": ScenarioConfig(
        name="Realistic Count Data",
        ectopic_type=EctopicType.NOISY_COPY,
        intrinsic_type=IntrinsicType.MEDIUM_EFFECT,
        spatial_structure=SpatialStructure.MODULES,
        ectopic_noise_level=0.1,
        intrinsic_boost_range=(2.0, 5.0),
    ),

    "cell_type_based": ScenarioConfig(
        name="Cell Type Based",
        ectopic_type=EctopicType.MARKER_SWAP,
        intrinsic_type=IntrinsicType.STRESS_MODULE,
        spatial_structure=SpatialStructure.CELL_TYPES,
    ),

    "hardest": ScenarioConfig(
        name="Hardest Scenario",
        ectopic_type=EctopicType.PARTIAL_MIX,
        intrinsic_type=IntrinsicType.SMALL_EFFECT,
        spatial_structure=SpatialStructure.CELL_TYPES,
        ectopic_mix_alpha=0.5,
        ectopic_noise_level=0.15,
        intrinsic_boost_range=(1.5, 3.0),
    ),
}


def generate_scenario_data(
    scenario: ScenarioConfig,
    n_spots: int = 3000,
    n_genes: int = 500,
    n_ectopic: int = 100,
    n_intrinsic: int = 300,
    min_distance_factor: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic data according to a scenario configuration.

    Pipeline: mu matrix -> inject anomalies on mu -> ZINB counts -> log1p normalize

    Returns
    -------
    X : np.ndarray
        Log1p-normalized count matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels : np.ndarray
        0=normal, 1=ectopic, 2=intrinsic
    ectopic_idx : np.ndarray
        Indices of ectopic anomalies
    intrinsic_idx : np.ndarray
        Indices of intrinsic anomalies
    metadata : dict
        Additional information (donor positions, cell types, etc.)
    """
    rng = np.random.RandomState(random_state)

    # Initialize cell_types - only populated for CELL_TYPES spatial structure
    cell_types = None

    # ==================================================================
    # Step 1: Generate spatial structure -> mu matrix
    # ==================================================================
    if scenario.spatial_structure == SpatialStructure.MODULES:
        mu_matrix, coords, coords_norm = _generate_module_mu(
            n_spots, n_genes, scenario.n_modules, rng
        )
    elif scenario.spatial_structure == SpatialStructure.CELL_TYPES:
        mu_matrix, coords, coords_norm, cell_types = _generate_cell_type_mu(
            n_spots, n_genes, scenario.n_cell_types, rng
        )
    elif scenario.spatial_structure == SpatialStructure.GRADIENTS:
        mu_matrix, coords, coords_norm = _generate_gradient_mu(
            n_spots, n_genes, rng
        )
    elif scenario.spatial_structure == SpatialStructure.LAYERED:
        mu_matrix, coords, coords_norm = _generate_layered_mu(
            n_spots, n_genes, rng
        )
    else:
        mu_matrix, coords, coords_norm = _generate_module_mu(
            n_spots, n_genes, scenario.n_modules, rng
        )

    # ==================================================================
    # Step 2: Apply library size variation
    # ==================================================================
    library_sizes = rng.lognormal(
        mean=np.log(scenario.library_size_mean),
        sigma=scenario.library_size_cv,
        size=(n_spots, 1),
    )
    current_totals = mu_matrix.sum(axis=1, keepdims=True) + 1e-8
    mu_matrix = mu_matrix * (library_sizes / current_totals)

    # ==================================================================
    # Step 3: Inject ectopic anomalies on mu
    # ==================================================================
    labels = np.zeros(n_spots, dtype=int)
    metadata = {"scenario": scenario.name}
    donor_positions = coords.copy()

    min_dist = min_distance_factor  # In normalized coordinates
    candidate_ectopic = rng.choice(n_spots, min(n_ectopic, n_spots), replace=False)
    successful_ectopic = []

    # CRITICAL FIX: Copy mu_matrix BEFORE the loop to avoid cascading contamination
    # Without this, modified ectopic spots could become donors for later ectopics
    mu_original = mu_matrix.copy()

    for idx in candidate_ectopic:
        distances = np.linalg.norm(coords_norm - coords_norm[idx], axis=1)
        distant_spots = np.where(distances > min_dist)[0]

        if len(distant_spots) > 0:
            donor = rng.choice(distant_spots)

            # Use mu_original to avoid cascading contamination
            if scenario.ectopic_type == EctopicType.EXACT_COPY:
                mu_matrix[idx] = mu_original[donor].copy()

            elif scenario.ectopic_type == EctopicType.NOISY_COPY:
                noise = rng.normal(
                    1.0, scenario.ectopic_noise_level, n_genes
                )
                noise = np.maximum(noise, 0.1)  # Ensure positive
                mu_matrix[idx] = mu_original[donor] * noise

            elif scenario.ectopic_type == EctopicType.PARTIAL_MIX:
                alpha = scenario.ectopic_mix_alpha
                mu_matrix[idx] = alpha * mu_original[donor] + (1 - alpha) * mu_original[idx]
                if scenario.ectopic_noise_level > 0:
                    noise = rng.normal(1.0, scenario.ectopic_noise_level, n_genes)
                    noise = np.maximum(noise, 0.1)
                    mu_matrix[idx] *= noise

            elif scenario.ectopic_type == EctopicType.GRADIENT_MIX:
                max_dist = np.sqrt(2)  # Max distance in normalized coords
                dist_ratio = distances[donor] / max_dist
                alpha = 0.5 + 0.4 * dist_ratio
                mu_matrix[idx] = alpha * mu_original[donor] + (1 - alpha) * mu_original[idx]

            elif scenario.ectopic_type == EctopicType.MARKER_SWAP:
                if scenario.spatial_structure == SpatialStructure.CELL_TYPES:
                    current_type = cell_types[idx]
                    donor_type = cell_types[donor]
                    if current_type != donor_type:
                        genes_per_type = n_genes // scenario.n_cell_types
                        my_markers = slice(
                            current_type * genes_per_type,
                            (current_type + 1) * genes_per_type,
                        )
                        donor_markers = slice(
                            donor_type * genes_per_type,
                            (donor_type + 1) * genes_per_type,
                        )
                        mu_matrix[idx, my_markers] = mu_original[donor, my_markers]
                        mu_matrix[idx, donor_markers] = mu_original[donor, donor_markers]
                else:
                    mu_matrix[idx] = mu_original[donor].copy()

            labels[idx] = 1
            donor_positions[idx] = coords[donor]
            successful_ectopic.append(idx)

    ectopic_idx = np.array(successful_ectopic)

    if len(ectopic_idx) < n_ectopic:
        warnings.warn(f"Only {len(ectopic_idx)}/{n_ectopic} ectopic anomalies injected")

    # ==================================================================
    # Step 4: Inject intrinsic anomalies on mu
    # ==================================================================
    remaining = np.setdiff1d(np.arange(n_spots), ectopic_idx)
    intrinsic_idx = rng.choice(remaining, min(n_intrinsic, len(remaining)), replace=False)

    boost_low, boost_high = scenario.intrinsic_boost_range

    for idx in intrinsic_idx:
        frac_low, frac_high = scenario.intrinsic_gene_fraction
        n_affected = rng.randint(
            int(n_genes * frac_low), int(n_genes * frac_high) + 1
        )
        affected_genes = rng.choice(n_genes, n_affected, replace=False)

        if scenario.intrinsic_type == IntrinsicType.STRESS_MODULE:
            # Activate last 10% of genes as "stress response"
            stress_genes = np.arange(n_genes - n_genes // 10, n_genes)
            boost_factors = rng.uniform(boost_low, boost_high, len(stress_genes))
            mu_matrix[idx, stress_genes] *= boost_factors
        else:
            # Upregulate affected genes
            boost_factors = rng.uniform(boost_low, boost_high, n_affected)
            mu_matrix[idx, affected_genes] *= boost_factors

            # Downregulate some other genes
            n_down = n_affected // 3
            down_genes = rng.choice(
                np.setdiff1d(np.arange(n_genes), affected_genes),
                min(n_down, n_genes - n_affected),
                replace=False,
            )
            mu_matrix[idx, down_genes] *= rng.uniform(0.1, 0.3, len(down_genes))

        labels[idx] = 2

    # ==================================================================
    # Step 5: Generate ZINB counts
    # ==================================================================
    r = scenario.dispersion
    p = r / (r + mu_matrix + 1e-12)
    X_counts = rng.negative_binomial(n=r, p=p)

    # Expression-dependent dropout
    gene_means = mu_matrix.mean(axis=0)
    gene_dropout_rates = scenario.dropout_rate * np.exp(
        -gene_means / (gene_means.mean() + 1e-10)
    )
    gene_dropout_rates = np.clip(gene_dropout_rates, 0.1, 0.8)

    dropout_mask = rng.random(X_counts.shape) < gene_dropout_rates
    X_counts[dropout_mask] = 0

    # Return raw counts (log1p is applied in prepare_data)
    X = X_counts.astype(float)

    # ==================================================================
    # Step 6: Build metadata
    # ==================================================================
    metadata["donor_positions"] = donor_positions
    metadata["ectopic_type"] = scenario.ectopic_type.value
    metadata["intrinsic_type"] = scenario.intrinsic_type.value
    if scenario.spatial_structure == SpatialStructure.CELL_TYPES:
        metadata["cell_types"] = cell_types

    return X, coords, labels, ectopic_idx, intrinsic_idx, metadata


# ==================================================================
# Spatial structure generators (produce mu matrices)
# ==================================================================

def _generate_module_mu(n_spots, n_genes, n_modules, rng):
    """Generate mu matrix with radial spatial modules."""
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    coords_norm = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-8
    )

    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.zeros((n_spots, n_genes))
    genes_per_module = n_genes // n_modules

    for m in range(n_modules):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-(distances ** 2) / 0.1)

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
        spatial_weight = np.exp(-(distances ** 2) / rng.uniform(0.05, 0.2))
        mu_matrix[:, g] = gene_base_rates[g] * (1 + rng.uniform(0.5, 2.0) * spatial_weight)

    return mu_matrix, coords, coords_norm


def _generate_cell_type_mu(n_spots, n_genes, n_cell_types, rng):
    """Generate mu matrix with discrete cell type regions."""
    from scipy.spatial.distance import cdist

    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    coords_norm = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-8
    )

    # Cell type centers (in normalized space)
    centers = rng.rand(n_cell_types, 2)
    dists = cdist(coords_norm, centers)
    cell_types = np.argmin(dists, axis=1)

    # Base gene expression rates
    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.tile(gene_base_rates, (n_spots, 1))

    # Cell type markers: boost expression for type-specific genes
    genes_per_type = n_genes // n_cell_types
    for t in range(n_cell_types):
        start = t * genes_per_type
        end = (t + 1) * genes_per_type
        idx = np.where(cell_types == t)[0]
        boost = rng.uniform(3.0, 8.0, size=(len(idx), end - start))
        mu_matrix[np.ix_(idx, range(start, end))] *= boost

    return mu_matrix, coords, coords_norm, cell_types


def _generate_gradient_mu(n_spots, n_genes, rng):
    """Generate mu matrix with continuous spatial gradients."""
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    coords_norm = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-8
    )

    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.zeros((n_spots, n_genes))

    for g in range(n_genes):
        direction = rng.rand(2) * 2 - 1
        direction /= np.linalg.norm(direction) + 1e-8
        projection = coords_norm @ direction
        projection = (projection - projection.min()) / (projection.max() - projection.min() + 1e-8)
        spatial_effect = rng.uniform(0.5, 3.0)
        mu_matrix[:, g] = gene_base_rates[g] * (1 + spatial_effect * projection)

    return mu_matrix, coords, coords_norm


def _generate_layered_mu(n_spots, n_genes, rng):
    """Generate mu matrix with layered spatial expression (like cortical layers)."""
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    coords_norm = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-8
    )

    # Layers based on normalized y coordinate
    n_layers = 6
    layer_boundaries = np.linspace(0, 1, n_layers + 1)
    y_norm = coords_norm[:, 1]
    layers = np.digitize(y_norm, layer_boundaries) - 1
    layers = np.clip(layers, 0, n_layers - 1)

    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.tile(gene_base_rates, (n_spots, 1))

    genes_per_layer = n_genes // n_layers
    for l in range(n_layers):
        layer_mask = layers == l
        start = l * genes_per_layer
        end = (l + 1) * genes_per_layer
        boost = rng.uniform(3.0, 8.0, size=(layer_mask.sum(), end - start))
        mu_matrix[np.ix_(np.where(layer_mask)[0], range(start, end))] *= boost

    return mu_matrix, coords, coords_norm


def list_scenarios():
    """List all available scenarios."""
    print("Available Scenarios:")
    print("=" * 60)
    for key, config in SCENARIOS.items():
        print(f"\n{key}: {config.name}")
        print(f"  Ectopic: {config.ectopic_type.value}")
        print(f"  Intrinsic: {config.intrinsic_type.value} (boost: {config.intrinsic_boost_range})")
        print(f"  Spatial: {config.spatial_structure.value}")
        print(f"  ZINB: dispersion={config.dispersion}, dropout={config.dropout_rate}")


if __name__ == "__main__":
    list_scenarios()
