"""High-level CASTOR API."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from castor._utils import set_seed
from castor.config import CASTORConfig
from castor.detection.registry import IntrinsicDetectorRegistry
from castor.detection.scoring import compute_diagnosis, compute_mad_zscore
from castor.model.inverse_prediction import InversePredictionModel
from castor.model.training import compute_scores, train_model
from castor.preprocessing.normalize import prepare_data, prepare_from_anndata

logger = logging.getLogger(__name__)


class CASTOR:
    """Dual-axis anomaly detection for spatial transcriptomics.

    Detects two orthogonal types of anomalies:

    - **Contextual** (ectopic): expression that doesn't belong at its spatial
      location, identified via inverse prediction (expression → position).
    - **Intrinsic**: globally unusual expression patterns, identified via a
      pluggable detector (default: PCA reconstruction error).

    Parameters
    ----------
    config : CASTORConfig | None
        Full configuration.  Individual keyword arguments override fields
        of the default config.
    **kwargs
        Passed to :class:`CASTORConfig`.

    Examples
    --------
    >>> from castor import CASTOR
    >>> c = CASTOR(intrinsic_method="pca_error")
    >>> results = c.fit_predict("data.h5ad")
    """

    def __init__(self, config: CASTORConfig | None = None, **kwargs: Any):
        cfg = config if config is not None else CASTORConfig()

        # Override defaults with kwargs
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise TypeError(f"Unknown parameter: {key!r}")

        self.config = cfg

        # Populated after fit_predict
        self.results_: pd.DataFrame | None = None
        self.coords_: np.ndarray | None = None
        self.X_norm_: np.ndarray | None = None
        self.gene_names_: list[str] | None = None
        self.scores_: dict[str, Any] | None = None
        self.params_: dict[str, Any] = asdict(cfg)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        input: str | Path | np.ndarray | Any,  # noqa: A002
        coords: np.ndarray | None = None,
        gene_names: list[str] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run the full detection pipeline.

        Parameters
        ----------
        input : str | Path | np.ndarray | AnnData
            - Path to ``.h5ad`` file.
            - Numpy expression matrix ``[n_spots, n_genes]`` (requires
              *coords*).
            - An ``AnnData`` object with spatial coordinates.
        coords : np.ndarray | None
            Spatial coordinates (required for numpy input).
        gene_names : list[str] | None
            Gene names.
        verbose : bool
            Log progress.

        Returns
        -------
        pd.DataFrame
            Columns: ``Local_Z``, ``Global_Z``, ``Diagnosis``, ``Confidence``.
        """
        cfg = self.config
        device = cfg.resolve_device()
        set_seed(cfg.random_state)

        # 1. Load / prepare data
        data = self._prepare_input(input, coords, gene_names, device)
        self.coords_ = data["coords_raw"]
        self.X_norm_ = data["X_norm"]
        self.gene_names_ = data.get("gene_names")

        if verbose:
            logger.info(
                "Data: %d spots, %d genes, device=%s",
                data["n_spots"],
                data["n_genes"],
                device,
            )

        # 2. Train inverse prediction model
        model = InversePredictionModel(
            in_dim=data["n_genes"],
            hid_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(device)

        model = train_model(
            model,
            data["x_tensor"],
            data["coords_tensor"],
            data["edge_index"],
            n_epochs=cfg.n_epochs,
            lr=cfg.learning_rate,
            lambda_pos=cfg.lambda_pos,
            lambda_self=cfg.lambda_self,
            verbose=verbose,
        )

        # 3. Compute scores
        scores = compute_scores(
            model,
            data["x_tensor"],
            data["coords_tensor"],
            data["edge_index"],
            random_state=cfg.random_state,
        )
        self.scores_ = scores

        # 4. Contextual axis: position prediction error → z-score
        z_local = compute_mad_zscore(scores["s_pos"])

        # 5. Intrinsic axis: pluggable detector → z-score
        intrinsic_scores = IntrinsicDetectorRegistry.compute(cfg.intrinsic_method, data["X_norm"])
        z_global = compute_mad_zscore(intrinsic_scores)

        # 6. Diagnosis
        results = compute_diagnosis(
            z_local,
            z_global,
            threshold_local=cfg.threshold_local,
            threshold_global=cfg.threshold_global,
        )

        self.results_ = results
        return results

    # ------------------------------------------------------------------
    # Downstream helpers
    # ------------------------------------------------------------------

    def get_contributing_genes(
        self,
        diagnosis_type: str = "Confirmed Anomaly",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Identify genes driving a specific anomaly type.

        Must be called after :meth:`fit_predict`.
        """
        self._check_fitted()
        from castor.enrichment.gene_analysis import identify_contributing_genes

        return identify_contributing_genes(
            self.X_norm_,
            self.results_,
            gene_names=self.gene_names_,
            diagnosis_type=diagnosis_type,
            top_n=top_n,
        )

    def plot_results(self, save_path: str | None = None, **kwargs: Any) -> None:
        """Generate the six-panel CASTOR results figure.

        Must be called after :meth:`fit_predict`.
        """
        self._check_fitted()
        from castor.visualization.plots import plot_castor_results

        plot_castor_results(
            self.coords_,
            self.results_,
            save_path=save_path,
            threshold_local=self.config.threshold_local,
            threshold_global=self.config.threshold_global,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        input: str | Path | np.ndarray | Any,  # noqa: A002
        coords: np.ndarray | None,
        gene_names: list[str] | None,
        device: str,
    ) -> dict[str, Any]:
        cfg = self.config

        # AnnData object
        if hasattr(input, "obsm"):
            data = prepare_from_anndata(
                input, n_top_genes=cfg.n_top_genes, k=cfg.k_neighbors, device=device
            )
            if "spatial" in input.obsm:
                data["coords_raw"] = np.asarray(input.obsm["spatial"])
            elif "X_spatial" in input.obsm:
                data["coords_raw"] = np.asarray(input.obsm["X_spatial"])
            else:
                data["coords_raw"] = data["coords_norm"]
            data["gene_names"] = gene_names
            return data

        # File path
        if isinstance(input, (str, Path)):
            from castor.io.loaders import auto_load

            X, raw_coords, gnames = auto_load(input, coords_path=None if coords is None else None)
            # If input is h5ad, auto_load handles coords
            if str(input).endswith(".h5ad"):
                X, raw_coords, gnames = auto_load(input)
            else:
                if coords is None:
                    raise ValueError("coords is required for CSV input")
                # coords provided as path or array
                if isinstance(coords, (str, Path)):
                    from castor.io.loaders import load_csv

                    X, raw_coords, gnames = load_csv(input, coords)
                else:
                    raw_coords = coords
                    import pandas as pd

                    expr_df = pd.read_csv(str(input), index_col=0)
                    X = expr_df.values.astype(np.float64)
                    gnames = expr_df.columns.tolist()

            data = prepare_data(
                X,
                raw_coords,
                k=cfg.k_neighbors,
                device=device,
                log_transform=cfg.log_transform,
                scale=cfg.scale,
            )
            data["coords_raw"] = raw_coords
            data["gene_names"] = gene_names or gnames
            return data

        # Numpy array
        if isinstance(input, np.ndarray):
            if coords is None:
                raise ValueError("coords is required when input is a numpy array")
            data = prepare_data(
                input,
                coords,
                k=cfg.k_neighbors,
                device=device,
                log_transform=cfg.log_transform,
                scale=cfg.scale,
            )
            data["coords_raw"] = coords
            data["gene_names"] = gene_names
            return data

        raise TypeError(f"Unsupported input type: {type(input)}")

    def _check_fitted(self) -> None:
        if self.results_ is None:
            raise RuntimeError("Call fit_predict() first.")
