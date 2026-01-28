"""Publication-quality visualisation for CASTOR results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DIAGNOSIS_COLORS: dict[str, str] = {
    "Normal": "lightgrey",
    "Contextual Anomaly": "orange",
    "Intrinsic Anomaly": "green",
    "Confirmed Anomaly": "red",
}


def plot_reliability_matrix(
    results: pd.DataFrame,
    ax: plt.Axes | None = None,
    s: int = 20,
    alpha: float = 0.6,
    threshold_local: float = 2.0,
    threshold_global: float = 2.0,
) -> plt.Axes:
    """Local_Z vs Global_Z scatter with threshold lines.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain ``Local_Z``, ``Global_Z``, ``Diagnosis``.
    ax : plt.Axes | None
        Axes to draw on (created if *None*).
    s : int
        Point size.
    alpha : float
        Transparency.
    threshold_local, threshold_global : float
        Threshold lines.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(
        data=results,
        x="Global_Z",
        y="Local_Z",
        hue="Diagnosis",
        style="Diagnosis",
        palette=DIAGNOSIS_COLORS,
        s=s,
        alpha=alpha,
        ax=ax,
    )

    ax.axvline(threshold_global, color="k", linestyle="--", alpha=0.5)
    ax.axhline(threshold_local, color="k", linestyle="--", alpha=0.5)
    ax.set_title("Reliability Matrix", fontsize=10)
    ax.set_xlabel("Global Score (Intrinsic Anomaly)")
    ax.set_ylabel("Local Score (Contextual Anomaly)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    return ax


def plot_castor_results(
    coords: np.ndarray,
    results: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple[float, float] = (24, 16),
    dpi: int = 300,
    threshold_local: float = 2.0,
    threshold_global: float = 2.0,
) -> None:
    """Generate the six-panel CASTOR results figure.

    Panels
    ------
    A. Spatial layout
    B. Final anomaly score (max of local and global)
    C. Diagnostic categories
    D. Local scores (Reds)
    E. Global scores (Blues)
    F. Reliability matrix

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    results : pd.DataFrame
        CASTOR results with ``Local_Z``, ``Global_Z``, ``Diagnosis``.
    save_path : str | None
        File path to save the figure.
    figsize : tuple
        Figure size.
    dpi : int
        Resolution.
    threshold_local, threshold_global : float
        Threshold lines for the reliability matrix.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # A -- Spatial layout
    axes[0].scatter(coords[:, 0], coords[:, 1], s=0.5, c="lightgrey", alpha=0.6)
    axes[0].set_title("A. Spatial Layout", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("spatial1")
    axes[0].set_ylabel("spatial2")
    axes[0].set_aspect("equal", adjustable="box")

    # B -- Final anomaly score
    final = np.maximum(results["Local_Z"].values, results["Global_Z"].values)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], s=0.5, c=final, cmap="viridis", alpha=0.6)
    axes[1].set_title("B. CASTOR Final Anomaly Score", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("spatial1")
    axes[1].set_ylabel("spatial2")
    axes[1].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[1], shrink=0.6)

    # C -- Diagnosis
    for diag, color in DIAGNOSIS_COLORS.items():
        mask = results["Diagnosis"] == diag
        if mask.sum() > 0:
            axes[2].scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=0.5,
                c=color,
                label=f"{diag} ({mask.sum()})",
                alpha=0.6,
            )
    axes[2].set_title("C. Diagnostic Categories", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("spatial1")
    axes[2].set_ylabel("spatial2")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # D -- Local scores
    vmax_local = np.percentile(results["Local_Z"], 95)
    sc = axes[3].scatter(
        coords[:, 0],
        coords[:, 1],
        s=0.5,
        c=results["Local_Z"],
        cmap="Reds",
        vmin=-2,
        vmax=vmax_local,
        alpha=0.6,
    )
    axes[3].set_title("D. Local Score (Contextual Anomaly)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("spatial1")
    axes[3].set_ylabel("spatial2")
    axes[3].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[3], shrink=0.6)

    # E -- Global scores
    vmax_global = np.percentile(results["Global_Z"], 95)
    sc = axes[4].scatter(
        coords[:, 0],
        coords[:, 1],
        s=0.5,
        c=results["Global_Z"],
        cmap="Blues",
        vmin=-2,
        vmax=vmax_global,
        alpha=0.6,
    )
    axes[4].set_title("E. Global Score (Intrinsic Anomaly)", fontsize=10, fontweight="bold")
    axes[4].set_xlabel("spatial1")
    axes[4].set_ylabel("spatial2")
    axes[4].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[4], shrink=0.6)

    # F -- Reliability matrix
    sns.scatterplot(
        data=results,
        x="Global_Z",
        y="Local_Z",
        hue="Diagnosis",
        style="Diagnosis",
        palette=DIAGNOSIS_COLORS,
        s=20,
        alpha=0.6,
        ax=axes[5],
    )
    axes[5].axvline(threshold_global, color="k", linestyle="--", alpha=0.5)
    axes[5].axhline(threshold_local, color="k", linestyle="--", alpha=0.5)
    axes[5].set_title("F. Reliability Matrix", fontsize=10, fontweight="bold")
    axes[5].set_xlabel("Global Score (Intrinsic Anomaly)")
    axes[5].set_ylabel("Local Score (Contextual Anomaly)")
    axes[5].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)

    plt.close(fig)


# ------------------------------------------------------------------
# Enrichment visualisation
# ------------------------------------------------------------------

def _truncate(text: str, maxlen: int = 50) -> str:
    """Truncate long pathway names for axis labels."""
    return text if len(text) <= maxlen else text[: maxlen - 1] + "\u2026"


def plot_contributing_genes(
    contrib: pd.DataFrame,
    save_path: str | None = None,
    top_n: int = 20,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
) -> None:
    """Horizontal bar chart of top contributing genes.

    Parameters
    ----------
    contrib : pd.DataFrame
        Output of :func:`identify_contributing_genes`.
        Must have ``Gene`` and ``Fold_Change`` columns.
    save_path : str | None
        File path to save the figure.
    top_n : int
        Number of genes to show.
    figsize : tuple | None
        Figure size (auto-scaled if *None*).
    dpi : int
        Resolution.
    """
    df = contrib.head(top_n).copy()
    if df.empty:
        return
    df = df.iloc[::-1]  # reverse so top gene is at top

    if figsize is None:
        figsize = (7, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["Fold_Change"]]
    ax.barh(df["Gene"], df["Fold_Change"], color=colors, edgecolor="none", height=0.7)

    ax.set_xlabel("Fold Change (anomaly vs normal)")
    ax.set_title("Contributing Genes", fontsize=11, fontweight="bold")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_enrichment_bar(
    enrichment: pd.DataFrame,
    save_path: str | None = None,
    top_n: int = 20,
    title: str = "Pathway Enrichment",
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
) -> None:
    """Horizontal bar chart of top enriched pathways.

    Parameters
    ----------
    enrichment : pd.DataFrame
        Output of :func:`run_pathway_enrichment`.
        Must have ``Pathway``, ``Enrichment_Score``, ``Q_value``, ``Gene_Count``.
    save_path : str | None
        File path to save the figure.
    top_n : int
        Number of pathways to show.
    title : str
        Figure title.
    figsize : tuple | None
        Figure size (auto-scaled if *None*).
    dpi : int
        Resolution.
    """
    if enrichment.empty:
        return

    # Select top pathways by absolute enrichment score
    df = enrichment.copy()
    df["abs_NES"] = df["Enrichment_Score"].abs()
    df = df.nlargest(top_n, "abs_NES").drop(columns=["abs_NES"])
    df = df.iloc[::-1]  # reverse for top-down ordering

    if figsize is None:
        figsize = (8, 0.4 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["Enrichment_Score"]]
    bars = ax.barh(
        [_truncate(p) for p in df["Pathway"]],
        df["Enrichment_Score"],
        color=colors,
        edgecolor="none",
        height=0.7,
    )

    # Annotate gene count
    for bar, gc in zip(bars, df["Gene_Count"]):
        w = bar.get_width()
        offset = 0.05 * max(abs(df["Enrichment_Score"].max()), abs(df["Enrichment_Score"].min()))
        xpos = w + offset if w >= 0 else w - offset
        ha = "left" if w >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"n={int(gc)}",
                va="center", ha=ha, fontsize=7, color="grey")

    ax.set_xlabel("Normalised Enrichment Score (NES)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_enrichment_dotplot(
    enrichment: pd.DataFrame,
    save_path: str | None = None,
    top_n: int = 20,
    title: str = "Pathway Enrichment",
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
) -> None:
    """Dot plot of enriched pathways.

    X-axis is enrichment score, dot size encodes gene count,
    and dot colour encodes statistical significance (-log10 Q-value).

    Parameters
    ----------
    enrichment : pd.DataFrame
        Must have ``Pathway``, ``Enrichment_Score``, ``Q_value``, ``Gene_Count``.
    save_path : str | None
        File path to save the figure.
    top_n : int
        Number of pathways to show.
    title : str
        Figure title.
    figsize : tuple | None
        Figure size (auto-scaled if *None*).
    dpi : int
        Resolution.
    """
    if enrichment.empty:
        return

    df = enrichment.copy()
    df["abs_NES"] = df["Enrichment_Score"].abs()
    df = df.nlargest(top_n, "abs_NES").drop(columns=["abs_NES"])
    df = df.iloc[::-1]

    df["neg_log10_q"] = -np.log10(df["Q_value"].clip(lower=1e-10))

    if figsize is None:
        figsize = (8, 0.4 * len(df) + 2)
    fig, ax = plt.subplots(figsize=figsize)

    size_scale = 200 / max(df["Gene_Count"].max(), 1)
    sc = ax.scatter(
        df["Enrichment_Score"],
        [_truncate(p) for p in df["Pathway"]],
        s=df["Gene_Count"] * size_scale,
        c=df["neg_log10_q"],
        cmap="YlOrRd",
        edgecolors="k",
        linewidths=0.3,
        alpha=0.85,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("$-\\log_{10}$(Q-value)", fontsize=9)

    # Size legend
    gc_vals = df["Gene_Count"]
    legend_sizes = np.unique(np.round(np.linspace(gc_vals.min(), gc_vals.max(), 4)).astype(int))
    for gs in legend_sizes:
        ax.scatter([], [], s=gs * size_scale, c="grey", edgecolors="k",
                   linewidths=0.3, label=str(gs))
    ax.legend(title="Gene Count", loc="lower right", fontsize=7, title_fontsize=8,
              framealpha=0.9)

    ax.set_xlabel("Normalised Enrichment Score (NES)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
