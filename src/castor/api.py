"""High-level CASTOR API."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_lncrna_like(gene_name: str) -> bool:
    """Check if a gene name looks like lncRNA or non-coding RNA.

    Patterns detected:
    - Human: AL/AC + numbers (e.g. AL157935.1, AC234031.1)
    - Human: LINC genes (e.g. LINC01862)
    - Mouse: Gm + numbers (e.g. Gm10530, Gm14066)
    - Mouse: Rik genes (e.g. 9530034E10Rik, A530041M06Rik)
    """
    # Human lncRNA patterns
    if re.match(r'^(AL|AC)\d+', gene_name):
        return True
    if re.match(r'^LINC\d+', gene_name):
        return True
    # Mouse lncRNA patterns
    if re.match(r'^Gm\d+', gene_name):
        return True
    if re.search(r'Rik$', gene_name):  # Ends with Rik
        return True
    return False

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
        self.model_: InversePredictionModel | None = None
        self.data_: dict[str, Any] | None = None
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
        self.model_ = model
        self.data_ = data

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

    def explain_spots(
        self,
        spot_indices: list[int] | None = None,
        top_n_spots: int = 10,
        top_n_genes: int = 10,
        diagnosis_type: str = "Contextual Anomaly",
        exclude_lncrna: bool = False,
        method: str = "gradient",
        n_steps: int = 50,
    ) -> dict[str, Any]:
        """Explain which genes contribute to position prediction for specific spots.

        Parameters
        ----------
        spot_indices : list[int] | None
            Specific spot indices to analyze. If None, uses top anomalies.
        top_n_spots : int
            Number of top anomaly spots to analyze (if spot_indices is None).
        top_n_genes : int
            Number of top genes to return per spot.
        diagnosis_type : str
            Filter spots by diagnosis type when spot_indices is None.
        exclude_lncrna : bool
            If True, exclude lncRNA-like genes from results. Patterns excluded:
            - Human: AL/AC + numbers (e.g. AL157935.1), LINC genes
            - Mouse: Gm + numbers (e.g. Gm10530), Rik genes (e.g. 9530034E10Rik)
            This helps focus on protein-coding genes for biological interpretation.
        method : str
            Attribution method: "gradient" (simple gradient) or "integrated_gradients".
            Integrated Gradients satisfies the completeness axiom:
            Σ attribution_i = f(x) - f(baseline), making attributions sum to output.
        n_steps : int
            Number of steps for Integrated Gradients approximation (default: 50).

        Returns
        -------
        dict
            Keys:
            - ``gene_summary``: DataFrame of genes ranked by mean |gradient|
            - ``spot_details``: List of per-spot DataFrames with gene attributions
            - ``spot_indices``: Indices of analyzed spots
        """
        self._check_fitted()
        import torch

        if self.model_ is None or self.data_ is None:
            raise RuntimeError("Model not available. Re-run fit_predict().")

        # Select spots to analyze
        if spot_indices is None:
            mask = self.results_["Diagnosis"] == diagnosis_type
            if mask.sum() == 0:
                logger.warning("No spots with diagnosis '%s'", diagnosis_type)
                return {"gene_summary": pd.DataFrame(), "spot_details": [], "spot_indices": []}
            spot_indices = (
                self.results_[mask]
                .nlargest(top_n_spots, "Local_Z")
                .index.tolist()
            )

        model = self.model_
        x_tensor = self.data_["x_tensor"]
        coords_tensor = self.data_["coords_tensor"]
        gene_names = self.gene_names_ or [f"gene_{i}" for i in range(x_tensor.shape[1])]

        model.eval()
        device = next(model.parameters()).device

        all_grads = []
        spot_details = []

        # Baseline for Integrated Gradients (zero vector)
        baseline = torch.zeros(1, x_tensor.shape[1], device=device)

        for spot_idx in spot_indices:
            x_spot = x_tensor[spot_idx : spot_idx + 1].clone().to(device)
            coord_spot = coords_tensor[spot_idx : spot_idx + 1].to(device)

            if method == "integrated_gradients":
                # Integrated Gradients: IG_i = (x_i - baseline_i) × ∫₀¹ ∂f/∂x_i dα
                # Approximate integral with Riemann sum
                integrated_grads_x = torch.zeros_like(x_spot)
                integrated_grads_y = torch.zeros_like(x_spot)

                for step in range(n_steps):
                    # Interpolate between baseline and input
                    alpha = step / n_steps
                    x_interp = baseline + alpha * (x_spot - baseline)
                    x_interp = x_interp.clone().detach().requires_grad_(True)

                    # Forward pass
                    h = model.encoder(x_interp)
                    pos_pred = model.pos_predictor(h)

                    # Gradient for X coordinate
                    model.zero_grad()
                    pos_pred[0, 0].backward(retain_graph=True)
                    integrated_grads_x += x_interp.grad.clone()

                    # Gradient for Y coordinate
                    x_interp.grad.zero_()
                    pos_pred[0, 1].backward()
                    integrated_grads_y += x_interp.grad.clone()

                # Average gradients and multiply by (input - baseline)
                integrated_grads_x = (integrated_grads_x / n_steps) * (x_spot - baseline)
                integrated_grads_y = (integrated_grads_y / n_steps) * (x_spot - baseline)

                grad_x = integrated_grads_x.detach().cpu().numpy().flatten()
                grad_y = integrated_grads_y.detach().cpu().numpy().flatten()
                grad = np.sqrt(grad_x**2 + grad_y**2)  # Magnitude

                # Store directional gradients for visualization
                _grad_x = grad_x
                _grad_y = grad_y

            else:  # Simple gradient
                x_spot.requires_grad_(True)

                # Forward pass
                h = model.encoder(x_spot)
                pos_pred = model.pos_predictor(h)

                # Position prediction error
                error = ((pos_pred - coord_spot) ** 2).sum()
                error.backward()

                grad = x_spot.grad.detach().cpu().numpy().flatten()
                _grad_x = None
                _grad_y = None

            all_grads.append(grad)

            # Compute predicted position for this spot
            with torch.no_grad():
                h = model.encoder(x_spot)
                pos_pred_final = model.pos_predictor(h)
                disp = (pos_pred_final - coord_spot).cpu().numpy().flatten()

                # For IG: also compute baseline prediction
                if method == "integrated_gradients":
                    h_baseline = model.encoder(baseline)
                    pos_baseline = model.pos_predictor(h_baseline)
                    pos_diff = (pos_pred_final - pos_baseline).cpu().numpy().flatten()

            # Per-spot detail
            if method == "integrated_gradients":
                spot_df = pd.DataFrame({
                    "gene": gene_names,
                    "attribution_x": _grad_x,
                    "attribution_y": _grad_y,
                    "attribution_mag": grad,
                    "expression": x_spot.detach().cpu().numpy().flatten(),
                }).sort_values("attribution_mag", ascending=False)
                # Verify completeness: sum of attributions should equal pos_diff
                sum_attr_x = _grad_x.sum()
                sum_attr_y = _grad_y.sum()
                spot_df.attrs["sum_attribution_x"] = sum_attr_x
                spot_df.attrs["sum_attribution_y"] = sum_attr_y
                spot_df.attrs["pos_diff_x"] = pos_diff[0]
                spot_df.attrs["pos_diff_y"] = pos_diff[1]
                spot_df.attrs["completeness_error_x"] = abs(sum_attr_x - pos_diff[0])
                spot_df.attrs["completeness_error_y"] = abs(sum_attr_y - pos_diff[1])
            else:
                spot_df = pd.DataFrame({
                    "gene": gene_names,
                    "gradient": grad,
                    "abs_gradient": np.abs(grad),
                    "expression": x_spot.detach().cpu().numpy().flatten(),
                }).sort_values("abs_gradient", ascending=False)

            # Filter lncRNA if requested
            if exclude_lncrna:
                spot_df = spot_df[~spot_df["gene"].apply(_is_lncrna_like)]

            spot_df.attrs["spot_idx"] = spot_idx
            spot_df.attrs["displacement_x"] = disp[0]
            spot_df.attrs["displacement_y"] = disp[1]
            spot_df.attrs["local_z"] = self.results_.loc[spot_idx, "Local_Z"]

            spot_details.append(spot_df.head(top_n_genes))

        # Gene-level summary
        all_grads = np.array(all_grads)
        if method == "integrated_gradients":
            gene_summary = pd.DataFrame({
                "gene": gene_names,
                "mean_attribution": all_grads.mean(axis=0),
                "mean_abs_attribution": np.abs(all_grads).mean(axis=0),
                "std_attribution": all_grads.std(axis=0),
            }).sort_values("mean_abs_attribution", ascending=False)
        else:
            gene_summary = pd.DataFrame({
                "gene": gene_names,
                "mean_abs_gradient": np.abs(all_grads).mean(axis=0),
                "mean_gradient": all_grads.mean(axis=0),
                "std_gradient": all_grads.std(axis=0),
            }).sort_values("mean_abs_gradient", ascending=False)

        # Filter lncRNA if requested
        if exclude_lncrna:
            gene_summary = gene_summary[~gene_summary["gene"].apply(_is_lncrna_like)]

        return {
            "gene_summary": gene_summary,
            "spot_details": spot_details,
            "spot_indices": spot_indices,
        }

    def plot_spot_explanation(
        self,
        spot_idx: int,
        save_path: str | None = None,
        top_n_genes: int = 15,
        rank: int | None = None,
        diagnosis_type: str | None = None,
        exclude_lncrna: bool = False,
        method: str = "gradient",
        n_steps: int = 50,
    ) -> None:
        """Visualize gene attribution for a single spot.

        Parameters
        ----------
        spot_idx : int
            Index of the spot to explain.
        save_path : str | None
            Path to save figure.
        top_n_genes : int
            Number of top genes to show.
        rank : int | None
            Rank of this spot among anomalies (for title).
        diagnosis_type : str | None
            Diagnosis type (for title).
        exclude_lncrna : bool
            If True, exclude lncRNA-like genes from visualization.
        method : str
            "gradient" or "integrated_gradients".
        n_steps : int
            Number of steps for IG.
        """
        self._check_fitted()
        import matplotlib.pyplot as plt
        from adjustText import adjust_text
        import torch

        if self.model_ is None or self.data_ is None:
            raise RuntimeError("Model not available. Re-run fit_predict().")

        model = self.model_
        x_tensor = self.data_["x_tensor"]
        coords_tensor = self.data_["coords_tensor"]
        gene_names = self.gene_names_ or [f"gene_{i}" for i in range(x_tensor.shape[1])]

        model.eval()
        device = next(model.parameters()).device

        x_spot = x_tensor[spot_idx : spot_idx + 1].clone().to(device)
        coord_spot = coords_tensor[spot_idx : spot_idx + 1].to(device)
        baseline = torch.zeros(1, x_tensor.shape[1], device=device)

        if method == "integrated_gradients":
            # Integrated Gradients
            ig_x = torch.zeros_like(x_spot)
            ig_y = torch.zeros_like(x_spot)

            for step in range(n_steps):
                alpha = step / n_steps
                x_interp = baseline + alpha * (x_spot - baseline)
                x_interp = x_interp.clone().detach().requires_grad_(True)

                h = model.encoder(x_interp)
                pos_pred = model.pos_predictor(h)

                model.zero_grad()
                pos_pred[0, 0].backward(retain_graph=True)
                ig_x += x_interp.grad.clone()

                x_interp.grad.zero_()
                pos_pred[0, 1].backward()
                ig_y += x_interp.grad.clone()

            # IG = (x - baseline) × avg_gradient
            attr_x = ((ig_x / n_steps) * (x_spot - baseline)).detach().cpu().numpy().flatten()
            attr_y = ((ig_y / n_steps) * (x_spot - baseline)).detach().cpu().numpy().flatten()
            attr_mag = np.sqrt(attr_x**2 + attr_y**2)

            # Compute f(x) - f(baseline) for verification
            with torch.no_grad():
                h = model.encoder(x_spot)
                pos_pred = model.pos_predictor(h)
                h_base = model.encoder(baseline)
                pos_base = model.pos_predictor(h_base)
                target_x = (pos_pred[0, 0] - pos_base[0, 0]).cpu().item()
                target_y = (pos_pred[0, 1] - pos_base[0, 1]).cpu().item()
        else:
            # Simple gradient
            x_spot_x = x_spot.clone().requires_grad_(True)
            h_x = model.encoder(x_spot_x)
            pos_x = model.pos_predictor(h_x)
            (pos_x[0, 0] - coord_spot[0, 0]).backward()
            grad_x = x_spot_x.grad.detach().cpu().numpy().flatten()

            x_spot_y = x_spot.clone().requires_grad_(True)
            h_y = model.encoder(x_spot_y)
            pos_y = model.pos_predictor(h_y)
            (pos_y[0, 1] - coord_spot[0, 1]).backward()
            grad_y = x_spot_y.grad.detach().cpu().numpy().flatten()

            expr = x_spot.detach().cpu().numpy().flatten()
            attr_x = grad_x * expr
            attr_y = grad_y * expr
            attr_mag = np.sqrt(grad_x**2 + grad_y**2)
            target_x, target_y = None, None

        # Get displacement for spatial plot
        with torch.no_grad():
            h = model.encoder(x_spot)
            pos_pred = model.pos_predictor(h)
            disp = (pos_pred - coord_spot).cpu().numpy().flatten()

        # Filter lncRNA if requested
        if exclude_lncrna:
            valid_mask = np.array([not _is_lncrna_like(g) for g in gene_names])
            valid_indices = np.where(valid_mask)[0]
            sorted_valid = valid_indices[np.argsort(attr_mag[valid_indices])[::-1]]
            top_idx = sorted_valid[:top_n_genes]
        else:
            top_idx = np.argsort(attr_mag)[-top_n_genes:][::-1]

        local_z = self.results_.loc[spot_idx, "Local_Z"]
        diagnosis = self.results_.loc[spot_idx, "Diagnosis"]

        # Build title
        rank_str = f"Rank #{rank} " if rank is not None else ""
        diag_str = f"[{diagnosis}]" if diagnosis_type is None else f"[{diagnosis_type}]"
        lncrna_str = " (excl. lncRNA)" if exclude_lncrna else ""
        method_str = " [IG]" if method == "integrated_gradients" else ""
        title = f"{rank_str}Spot {spot_idx} {diag_str} (Local_Z={local_z:.2f}){lncrna_str}{method_str}"

        if method == "integrated_gradients":
            # IG visualization: X/Y contribution bars
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Panel 1 (top-left): Total attribution magnitude
            ax1 = axes[0, 0]
            n_show = min(top_n_genes, len(top_idx))
            show_idx = top_idx[:n_show]
            ax1.barh(range(n_show), attr_mag[show_idx][::-1], color="steelblue", alpha=0.8)
            ax1.set_yticks(range(n_show))
            ax1.set_yticklabels([gene_names[i] for i in show_idx[::-1]], fontsize=9)
            ax1.set_xlabel("Attribution Magnitude")
            ax1.set_title(f"Top {n_show} Genes by |Attribution|")

            # Panel 2 (top-right): Spatial location
            ax2 = axes[0, 1]
            ax2.scatter(self.coords_[:, 0], self.coords_[:, 1], c="lightgray", s=5, alpha=0.3)
            ax2.scatter([self.coords_[spot_idx, 0]], [self.coords_[spot_idx, 1]],
                       c="red", s=100, marker="*", zorder=10, label="Actual")
            pred_pos = self.scores_["pos_pred"][spot_idx]
            coord_min, coord_max = self.coords_.min(axis=0), self.coords_.max(axis=0)
            pred_denorm = pred_pos * (coord_max - coord_min) + coord_min
            ax2.scatter([pred_denorm[0]], [pred_denorm[1]], c="blue", s=100, marker="x", zorder=10, label="Predicted")
            ax2.annotate("", xy=pred_denorm, xytext=self.coords_[spot_idx],
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))
            ax2.legend(loc="upper right")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_title("Spatial Location")
            ax2.set_aspect("equal")

            # Panel 3 (bottom-left): X-direction attributions
            ax3 = axes[1, 0]
            colors_x = ["#e74c3c" if v < 0 else "#3498db" for v in attr_x[show_idx][::-1]]
            ax3.barh(range(n_show), attr_x[show_idx][::-1], color=colors_x, alpha=0.8)
            ax3.set_yticks(range(n_show))
            ax3.set_yticklabels([gene_names[i] for i in show_idx[::-1]], fontsize=9)
            ax3.axvline(0, color="black", linewidth=0.5)
            ax3.set_xlabel("X Attribution")
            sum_x = attr_x.sum()
            ax3.set_title(f"X-direction Attribution\nΣ={sum_x:.4f}, Target={target_x:.4f}")

            # Panel 4 (bottom-right): Y-direction attributions
            ax4 = axes[1, 1]
            colors_y = ["#e74c3c" if v < 0 else "#27ae60" for v in attr_y[show_idx][::-1]]
            ax4.barh(range(n_show), attr_y[show_idx][::-1], color=colors_y, alpha=0.8)
            ax4.set_yticks(range(n_show))
            ax4.set_yticklabels([gene_names[i] for i in show_idx[::-1]], fontsize=9)
            ax4.axvline(0, color="black", linewidth=0.5)
            ax4.set_xlabel("Y Attribution")
            sum_y = attr_y.sum()
            ax4.set_title(f"Y-direction Attribution\nΣ={sum_y:.4f}, Target={target_y:.4f}")

        else:
            # Original gradient visualization
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Panel 1: Top genes bar chart
            ax1 = axes[0]
            ax1.barh(range(top_n_genes), attr_mag[top_idx][::-1], color="steelblue", alpha=0.8)
            ax1.set_yticks(range(top_n_genes))
            ax1.set_yticklabels([gene_names[i] for i in top_idx[::-1]], fontsize=9)
            ax1.set_xlabel("Gradient Magnitude")
            ax1.set_title(f"Top {top_n_genes} Genes")

            # Panel 2: Direction visualization
            ax2 = axes[1]
            n_show = min(10, len(top_idx))
            top_idx_show = top_idx[:n_show]
            angles = np.arctan2(attr_y[top_idx_show], attr_x[top_idx_show])
            contrib_mag = np.sqrt(attr_x[top_idx_show]**2 + attr_y[top_idx_show]**2)
            colors = plt.cm.tab10(np.arange(n_show))
            max_mag = contrib_mag.max() if contrib_mag.max() > 0 else 1
            sizes = (contrib_mag / max_mag) * 200 + 30

            for i in range(n_show):
                x_pos, y_pos = np.cos(angles[i]), np.sin(angles[i])
                ax2.scatter(x_pos, y_pos, s=sizes[i], c=[colors[i]], alpha=0.7,
                           edgecolors='black', linewidths=0.5, label=f"{gene_names[top_idx_show[i]]}")
                ax2.plot([0, x_pos], [0, y_pos], c=colors[i], alpha=0.3, lw=1)

            theta = np.linspace(0, 2*np.pi, 100)
            ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, lw=0.5)
            disp_norm = disp / (np.linalg.norm(disp) + 1e-8)
            ax2.annotate("", xy=(disp_norm[0]*0.7, disp_norm[1]*0.7), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=2, mutation_scale=15))
            ax2.text(disp_norm[0]*0.85, disp_norm[1]*0.85, "Disp", color="red", fontsize=9, fontweight="bold")
            ax2.axhline(0, color="gray", linestyle="--", alpha=0.2, linewidth=0.5)
            ax2.axvline(0, color="gray", linestyle="--", alpha=0.2, linewidth=0.5)
            ax2.set_xlim(-1.4, 1.4)
            ax2.set_ylim(-1.4, 1.4)
            ax2.set_xlabel("Contribution X")
            ax2.set_ylabel("Contribution Y")
            ax2.set_title("Gene Contributions (grad × expr)\nRed arrow = displacement")
            ax2.set_aspect("equal")
            ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)

            # Panel 3: Spatial context
            ax3 = axes[2]
            ax3.scatter(self.coords_[:, 0], self.coords_[:, 1], c="lightgray", s=5, alpha=0.3)
            ax3.scatter([self.coords_[spot_idx, 0]], [self.coords_[spot_idx, 1]],
                       c="red", s=100, marker="*", zorder=10, label="Actual")
            pred_pos = self.scores_["pos_pred"][spot_idx]
            coord_min, coord_max = self.coords_.min(axis=0), self.coords_.max(axis=0)
            pred_denorm = pred_pos * (coord_max - coord_min) + coord_min
            ax3.scatter([pred_denorm[0]], [pred_denorm[1]], c="blue", s=100, marker="x", zorder=10, label="Predicted")
            ax3.annotate("", xy=pred_denorm, xytext=self.coords_[spot_idx],
                        arrowprops=dict(arrowstyle="->", color="red", lw=2))
            ax3.legend(loc="upper right")
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_title("Spatial Location")
            ax3.set_aspect("equal")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", save_path)

        plt.close()

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
