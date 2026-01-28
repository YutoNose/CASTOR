"""
HER2ST breast cancer data loader for real-world benchmarking.

Loads HER2-positive breast cancer spatial transcriptomics data with
pathologist annotations as ground truth for anomaly detection evaluation.
"""

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


class HER2STDataLoader:
    """Load HER2ST breast cancer spatial transcriptomics data.

    Data source: https://zenodo.org/record/3957257
    Reference: Andersson et al. (2021) "Spatial Deconvolution of HER2 positive
               Breast Tumors Reveals Novel Intercellular Relationships"
    """

    def __init__(self, her2st_dir: str | Path):
        """Initialize data loader.

        Args:
            her2st_dir: Path to her2st data directory
        """
        self.her2st_dir = Path(her2st_dir)
        self.counts_dir = self.her2st_dir / "data" / "ST-cnts"
        self.spots_dir = self.her2st_dir / "data" / "ST-spotfiles"
        self.labels_dir = self.her2st_dir / "data" / "ST-pat" / "lbl"
        self.image_dir = self.her2st_dir / "data" / "ST-imgs"

        # Validate directory structure
        for d in [self.counts_dir, self.spots_dir, self.labels_dir]:
            if not d.exists():
                msg = f"Directory not found: {d}"
                raise FileNotFoundError(msg)

        # Get available sample IDs
        self.available_samples = self._get_available_samples()

    def _get_available_samples(self) -> list[str]:
        """Get list of available sample IDs with complete data."""
        samples = []

        # Check for samples with all required files
        for label_file in self.labels_dir.glob("*_labeled_coordinates.tsv"):
            sample_id = label_file.stem.replace("_labeled_coordinates", "")

            counts_file = self.counts_dir / f"{sample_id}.tsv.gz"
            spots_file = self.spots_dir / f"{sample_id}_selection.tsv"

            if counts_file.exists() and spots_file.exists():
                samples.append(sample_id)

        return sorted(samples)

    def load(
        self,
        sample_id: str,
        anomaly_labels: list[str] | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
        """Load a single sample with pathologist annotations.

        Args:
            sample_id: Sample identifier (e.g., 'A1', 'B1')
            anomaly_labels: List of pathologist labels to treat as anomalies.
                          Default: ['invasive cancer', 'dcis']

        Returns:
            Tuple of (X, coords, y_true, metadata)
            - X: Sparse count matrix (n_spots, n_genes)
            - coords: Spatial coordinates (n_spots, 2)
            - y_true: Binary anomaly labels (n_spots,)
            - metadata: Dictionary with sample information
        """
        if anomaly_labels is None:
            anomaly_labels = ["invasive cancer", "dcis"]

        if sample_id not in self.available_samples:
            msg = (
                f"Sample {sample_id} not found. "
                f"Available: {self.available_samples}"
            )
            raise ValueError(msg)

        # Load gene expression counts (spots x genes format)
        # Spot IDs in counts are array coords like "10x13"
        counts_file = self.counts_dir / f"{sample_id}.tsv.gz"
        with gzip.open(counts_file, "rt") as f:
            counts_df = pd.read_csv(f, sep="\t", index_col=0)

        # Load spot coordinates
        spots_file = self.spots_dir / f"{sample_id}_selection.tsv"
        spots_df = pd.read_csv(spots_file, sep="\t")

        # Create spot ID from array coords (ensure integer format to match counts/labels)
        spots_df["spot_id"] = (
            spots_df["x"].round().astype(int).astype(str) + "x" +
            spots_df["y"].round().astype(int).astype(str)
        )
        spots_df = spots_df.set_index("spot_id")

        # Load pathologist labels
        # Labels have Row.names as pixel-based IDs, and x/y as array coords (float)
        labels_file = self.labels_dir / f"{sample_id}_labeled_coordinates.tsv"
        labels_df = pd.read_csv(labels_file, sep="\t")

        # Drop rows with missing coordinates
        labels_df = labels_df.dropna(subset=["x", "y"])

        # Round label array coords to match spot array coords
        labels_df["array_x"] = labels_df["x"].round().astype(int)
        labels_df["array_y"] = labels_df["y"].round().astype(int)
        labels_df["spot_id"] = (
            labels_df["array_x"].astype(str) + "x" + labels_df["array_y"].astype(str)
        )

        # Set Row.names as index for labels
        labels_df = labels_df.set_index("Row.names")

        # Align data: only keep spots with labels
        common_spots = counts_df.index.intersection(labels_df["spot_id"])

        if len(common_spots) == 0:
            msg = f"No common spots found between counts and labels for {sample_id}"
            raise ValueError(msg)

        # Filter counts to common spots
        counts_df = counts_df.loc[common_spots]

        # Filter labels to common spots and reorder to match counts
        labels_df = labels_df[labels_df["spot_id"].isin(common_spots)]
        spot_id_to_rowname = dict(zip(labels_df["spot_id"], labels_df.index))
        row_order = [spot_id_to_rowname[sid] for sid in common_spots]
        labels_df = labels_df.loc[row_order]

        # Create expression matrix (n_spots, n_genes)
        X = counts_df.values
        X_sparse = sparse.csr_matrix(X)

        # Extract coordinates (use array coords x, y)
        coords = labels_df[["x", "y"]].values

        # Create binary anomaly labels
        y_true = np.zeros(len(common_spots), dtype=int)
        for label in anomaly_labels:
            # Use .values to convert pandas boolean mask to numpy for positional indexing
            mask = labels_df["label"].str.lower().str.contains(label, na=False).values
            y_true[mask] = 1

        # Get unique labels for metadata
        unique_labels = labels_df["label"].value_counts().to_dict()

        # Extract pixel coordinates from spots_df (more reliable than labels_df)
        pixel_coords = None
        if "pixel_x" in spots_df.columns and "pixel_y" in spots_df.columns:
            # Align spots_df to common_spots order
            aligned_spots = spots_df.reindex(common_spots)
            if aligned_spots["pixel_x"].notna().all():
                pixel_coords = aligned_spots[["pixel_x", "pixel_y"]].values

        # Load tissue image if available
        tissue_image = None
        if self.image_dir.exists():
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                img_path = self.image_dir / f"{sample_id}{ext}"
                if img_path.exists():
                    try:
                        from PIL import Image
                        tissue_image = np.array(Image.open(img_path))
                        break
                    except Exception as e:
                        print(f"Warning: Failed to load image {img_path}: {e}")

        # Metadata
        metadata = {
            "sample_id": sample_id,
            "n_spots": len(common_spots),
            "n_genes": counts_df.shape[1],
            "anomaly_labels": anomaly_labels,
            "unique_labels": unique_labels,
            "anomaly_fraction": y_true.sum() / len(y_true),
            "data_source": "HER2ST",
            "pixel_coords": pixel_coords,
            "tissue_image": tissue_image,
        }

        return X_sparse, coords, y_true, metadata

    def load_split(
        self,
        sample_id: str,
        normal_labels: list[str] | None = None,
        anomaly_labels: list[str] | None = None,
    ) -> tuple[
        sparse.csr_matrix,
        np.ndarray,
        sparse.csr_matrix,
        np.ndarray,
        dict[str, Any],
    ]:
        """Load a sample split into normal (reference) and full (target) data.

        This method is designed for competitor methods that require reference data:
        - STANDS: Use normal regions as reference, full sample as target
        - Sardine: Use normal regions as control, anomaly regions as disease

        Args:
            sample_id: Sample identifier
            normal_labels: List of labels to treat as normal (reference)
                          Default: ['connective tissue', 'fat', 'in situ']
            anomaly_labels: List of labels to treat as anomalies
                          Default: ['invasive cancer', 'dcis']

        Returns:
            Tuple of (X_normal, coords_normal, X_full, coords_full, metadata)
            - X_normal: Normal region expression (n_normal, n_genes)
            - coords_normal: Normal region coordinates (n_normal, 2)
            - X_full: Full sample expression (n_spots, n_genes)
            - coords_full: Full sample coordinates (n_spots, 2)
            - metadata: Dict with sample info including y_true labels
        """
        if normal_labels is None:
            # Default normal tissue types in breast
            normal_labels = ["connective tissue", "fat", "in situ"]

        if anomaly_labels is None:
            anomaly_labels = ["invasive cancer", "dcis"]

        # Load full sample
        X_full, coords_full, y_true, metadata = self.load(sample_id, anomaly_labels)

        # Get pathologist labels for filtering
        labels_file = self.labels_dir / f"{sample_id}_labeled_coordinates.tsv"
        labels_df = pd.read_csv(labels_file, sep="\t")
        labels_df = labels_df.dropna(subset=["x", "y"])
        labels_df["array_x"] = labels_df["x"].round().astype(int)
        labels_df["array_y"] = labels_df["y"].round().astype(int)
        labels_df["spot_id"] = (
            labels_df["array_x"].astype(str) + "x" + labels_df["array_y"].astype(str)
        )
        labels_df = labels_df.set_index("Row.names")

        # Get counts to align with labels
        counts_file = self.counts_dir / f"{sample_id}.tsv.gz"
        with gzip.open(counts_file, "rt") as f:
            counts_df = pd.read_csv(f, sep="\t", index_col=0)

        common_spots = counts_df.index.intersection(labels_df["spot_id"])
        labels_df = labels_df[labels_df["spot_id"].isin(common_spots)]
        spot_id_to_rowname = dict(zip(labels_df["spot_id"], labels_df.index))
        row_order = [spot_id_to_rowname[sid] for sid in common_spots]
        labels_df = labels_df.loc[row_order]

        # Create mask for normal spots
        normal_mask = np.zeros(len(common_spots), dtype=bool)
        for label in normal_labels:
            normal_mask |= labels_df["label"].str.lower().str.contains(label, na=False).values

        if normal_mask.sum() == 0:
            msg = f"No normal spots found with labels {normal_labels} in {sample_id}"
            raise ValueError(msg)

        # Extract normal region data
        X_normal = X_full[normal_mask]
        coords_normal = coords_full[normal_mask]

        # Update metadata
        metadata["normal_labels"] = normal_labels
        metadata["n_normal_spots"] = normal_mask.sum()
        metadata["normal_fraction"] = normal_mask.sum() / len(common_spots)

        return X_normal, coords_normal, X_full, coords_full, metadata

    def load_all_samples(
        self,
        anomaly_labels: list[str] | None = None,
        max_samples: int | None = None,
    ) -> dict[str, tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]]:
        """Load multiple samples.

        Args:
            anomaly_labels: List of pathologist labels to treat as anomalies
            max_samples: Maximum number of samples to load (for quick testing)

        Returns:
            Dictionary mapping sample_id to (X, coords, y_true, metadata)
        """
        samples_to_load = self.available_samples
        if max_samples is not None:
            samples_to_load = samples_to_load[:max_samples]

        results = {}
        for sample_id in samples_to_load:
            try:
                results[sample_id] = self.load(sample_id, anomaly_labels)
            except Exception as e:
                print(f"Warning: Failed to load {sample_id}: {e}")
                continue

        return results

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        return {
            "dataset": "HER2ST",
            "n_samples": len(self.available_samples),
            "available_samples": self.available_samples,
            "description": "HER2-positive breast cancer spatial transcriptomics",
            "reference": "Andersson et al. (2021)",
        }
