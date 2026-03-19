"""Tests pour src/config.py."""

import pytest
from src.config import load_config, get_column_name


def test_load_config():
    """La config se charge sans erreur et contient les sections attendues."""
    config = load_config()
    assert "project" in config
    assert "zone" in config
    assert "features" in config
    assert "clustering" in config
    assert "pipeline" in config


def test_config_has_codes_insee():
    """La config contient des codes INSEE pour la métropole."""
    config = load_config()
    codes = config["zone"]["codes_insee"]
    assert len(codes) >= 25  # 28 communes attendues
    assert "33063" in codes  # Bordeaux doit être présent


def test_config_clustering_algorithms():
    """La config définit au moins 4 algorithmes de clustering."""
    config = load_config()
    algos = config["clustering"]["algorithms"]
    assert len(algos) >= 4
    algo_names = [a["name"] for a in algos]
    assert "kmeans" in algo_names
    assert "hdbscan" in algo_names


def test_get_column_name_todo_raises():
    """get_column_name lève une erreur si le mapping vaut TODO."""
    config = load_config()
    # Les mappings sont initialement à TODO
    with pytest.raises(ValueError, match="TODO"):
        get_column_name(config, "surface_bat")


def test_get_column_name_missing_raises():
    """get_column_name lève une erreur si la feature n'existe pas."""
    config = load_config()
    with pytest.raises(ValueError, match="absente"):
        get_column_name(config, "feature_inexistante")
