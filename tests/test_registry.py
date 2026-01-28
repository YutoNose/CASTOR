"""Tests for the intrinsic detector registry."""

import numpy as np
import pytest


def test_builtin_methods_registered():
    from castor.detection.registry import IntrinsicDetectorRegistry

    methods = IntrinsicDetectorRegistry.list_methods()
    assert "pca_error" in methods
    assert "lof" in methods
    assert "isolation_forest" in methods
    assert "mahalanobis" in methods
    assert "ocsvm" in methods


def test_pca_error_returns_correct_shape():
    from castor.detection.registry import IntrinsicDetectorRegistry

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 50))
    scores = IntrinsicDetectorRegistry.compute("pca_error", X, n_components=10)
    assert scores.shape == (100,)
    assert np.all(np.isfinite(scores))


def test_register_custom_detector():
    from castor.detection.registry import IntrinsicDetectorRegistry, register_intrinsic_detector

    @register_intrinsic_detector("_test_custom")
    def my_detector(X, **kwargs):
        return np.ones(X.shape[0])

    assert "_test_custom" in IntrinsicDetectorRegistry.list_methods()
    scores = IntrinsicDetectorRegistry.compute("_test_custom", np.zeros((10, 5)))
    assert np.allclose(scores, 1.0)


def test_unknown_method_raises():
    from castor.detection.registry import IntrinsicDetectorRegistry

    with pytest.raises(KeyError, match="Unknown intrinsic detector"):
        IntrinsicDetectorRegistry.get("nonexistent_method_xyz")


def test_scoring_diagnosis():
    from castor.detection.scoring import compute_diagnosis

    z_local = np.array([0.0, 3.0, 0.5, 3.0])
    z_global = np.array([0.0, 0.5, 3.0, 3.0])
    df = compute_diagnosis(z_local, z_global)

    assert df.loc[0, "Diagnosis"] == "Normal"
    assert df.loc[1, "Diagnosis"] == "Contextual Anomaly"
    assert df.loc[2, "Diagnosis"] == "Intrinsic Anomaly"
    assert df.loc[3, "Diagnosis"] == "Confirmed Anomaly"


def test_model_forward():
    import torch

    from castor.model.graph import build_spatial_graph
    from castor.model.inverse_prediction import InversePredictionModel

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20)).astype(np.float32)
    coords = rng.standard_normal((50, 2)).astype(np.float32)

    edge_index = build_spatial_graph(coords, k=5)
    model = InversePredictionModel(in_dim=20, hid_dim=16, dropout=0.0)

    x_t = torch.tensor(X)
    h, pos_pred, x_self, x_neighbor = model(x_t, edge_index)

    assert h.shape == (50, 16)
    assert pos_pred.shape == (50, 2)
    assert x_self.shape == (50, 20)
    assert x_neighbor.shape == (50, 20)


def test_prepare_data():
    from castor.preprocessing.normalize import prepare_data

    rng = np.random.default_rng(1)
    X = rng.poisson(5, (80, 30)).astype(np.float64)
    coords = rng.standard_normal((80, 2))

    data = prepare_data(X, coords, k=5, device="cpu")
    assert data["n_spots"] == 80
    assert data["n_genes"] == 30
    assert data["x_tensor"].shape == (80, 30)
    assert data["edge_index"].shape[0] == 2
