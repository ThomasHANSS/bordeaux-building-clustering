"""Tests pour src/clustering.py."""

import numpy as np
import pytest

from src.clustering import evaluate_clustering, find_optimal_k, run_single_algorithm


def _make_test_data(n=500, k=3, random_state=42):
    """Crée des données synthétiques bien séparées."""
    rng = np.random.RandomState(random_state)
    centers = rng.rand(k, 5) * 10
    X = np.vstack([
        rng.randn(n // k, 5) + center
        for center in centers
    ])
    return X


def test_find_optimal_k():
    """find_optimal_k retourne un k dans la plage demandée."""
    X = _make_test_data(n=300, k=3)
    result = find_optimal_k(X, k_range=(2, 8), random_state=42)
    assert "k_elbow" in result
    assert "k_silhouette" in result
    assert 2 <= result["k_elbow"] <= 8
    assert 2 <= result["k_silhouette"] <= 8


def test_kmeans_returns_correct_shape():
    """K-Means retourne autant de labels que de samples."""
    X = _make_test_data(n=200)
    labels = run_single_algorithm(X, "kmeans", {}, n_clusters=3)
    assert len(labels) == 200
    assert len(set(labels)) <= 3


def test_evaluate_clustering_metrics():
    """evaluate_clustering retourne les 3 métriques attendues."""
    X = _make_test_data(n=200, k=3)
    labels = run_single_algorithm(X, "kmeans", {}, n_clusters=3)
    metrics = evaluate_clustering(X, labels)
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics
    assert "calinski_harabasz" in metrics
    assert -1 <= metrics["silhouette"] <= 1


def test_evaluate_handles_noise():
    """evaluate_clustering gère les labels -1 (bruit)."""
    X = _make_test_data(n=100, k=2)
    labels = np.array([0] * 40 + [1] * 40 + [-1] * 20)
    metrics = evaluate_clustering(X, labels)
    assert metrics["silhouette"] != -1  # Doit calculer sur les non-bruit
