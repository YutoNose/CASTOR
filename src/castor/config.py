"""Configuration dataclass for CASTOR."""

from dataclasses import dataclass


@dataclass
class CASTORConfig:
    """CASTOR hyperparameters.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension for the encoder.
    dropout : float
        Dropout rate.
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate.
    lambda_pos : float
        Weight for position prediction loss.
    lambda_self : float
        Weight for self-reconstruction loss.
    k_neighbors : int
        Number of spatial neighbors for the kNN graph.
    intrinsic_method : str
        Intrinsic anomaly detection method name (from registry).
    threshold_local : float
        Z-score threshold for contextual anomalies.
    threshold_global : float
        Z-score threshold for intrinsic anomalies.
    n_top_genes : int
        Number of highly variable genes to select from AnnData.
    log_transform : bool
        Apply log1p transformation.
    scale : bool
        Standardize expression per gene.
    device : str
        Device: "auto", "cpu", or "cuda".
    random_state : int
        Random seed.
    """

    hidden_dim: int = 64
    dropout: float = 0.3
    n_epochs: int = 100
    learning_rate: float = 1e-3
    lambda_pos: float = 0.5
    lambda_self: float = 1.0
    k_neighbors: int = 15
    intrinsic_method: str = "pca_error"
    threshold_local: float = 2.0
    threshold_global: float = 2.0
    n_top_genes: int = 2000
    log_transform: bool = True
    scale: bool = True
    device: str = "auto"
    random_state: int = 42

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
