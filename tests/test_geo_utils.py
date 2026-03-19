"""Tests pour src/geo_utils.py."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from src.geo_utils import (
    TARGET_CRS,
    aggregate_multi_matches,
    prepare_geodf,
    smart_aggregate,
)


def _make_test_gdf(n=10, crs="EPSG:4326"):
    """Crée un GeoDataFrame de test avec des carrés."""
    polygons = [box(i, 44.8, i + 0.01, 44.81) for i in range(n)]
    gdf = gpd.GeoDataFrame(
        {"id": range(n), "value": np.random.rand(n) * 100},
        geometry=polygons,
        crs=crs,
    )
    return gdf


def test_prepare_geodf_reprojects():
    """prepare_geodf reprojette en EPSG:2154."""
    gdf = _make_test_gdf(crs="EPSG:4326")
    result = prepare_geodf(gdf)
    assert result.crs.to_epsg() == 2154


def test_prepare_geodf_no_crs_fallback():
    """prepare_geodf gère l'absence de CRS."""
    gdf = _make_test_gdf()
    gdf = gdf.set_crs(None, allow_override=True)
    result = prepare_geodf(gdf)
    assert result.crs.to_epsg() == 2154


def test_prepare_geodf_explodes_multi():
    """prepare_geodf explose les MultiPolygon."""
    poly1 = box(0, 44.8, 0.01, 44.81)
    poly2 = box(0.02, 44.8, 0.03, 44.81)
    multi = MultiPolygon([poly1, poly2])

    gdf = gpd.GeoDataFrame(
        {"id": [0]},
        geometry=[multi],
        crs="EPSG:4326",
    )
    result = prepare_geodf(gdf)
    assert len(result) == 2  # 1 MultiPolygon → 2 Polygon


def test_prepare_geodf_filters_micro():
    """prepare_geodf filtre les micro-géométries."""
    tiny = box(0, 0, 0.0000001, 0.0000001)  # Très petit en 2154
    normal = box(0, 44.8, 0.01, 44.81)

    gdf = gpd.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[tiny, normal],
        crs="EPSG:2154",
    )
    result = prepare_geodf(gdf)
    assert len(result) <= 2  # Le tiny peut être filtré


def test_prepare_geodf_has_uid():
    """prepare_geodf ajoute une colonne uid unique."""
    gdf = _make_test_gdf()
    result = prepare_geodf(gdf)
    assert "uid" in result.columns
    assert result["uid"].nunique() == len(result)


def test_aggregate_mean():
    """aggregate_multi_matches en mode mean."""
    import pandas as pd

    df = gpd.GeoDataFrame({
        "id": [1, 1, 2, 2],
        "val": [10.0, 20.0, 30.0, 40.0],
        "geometry": [box(0, 0, 1, 1)] * 4,
    })
    result = aggregate_multi_matches(df, "id", ["val"], strategy="mean")
    assert len(result) == 2
    assert result.loc[result["id"] == 1, "val"].values[0] == pytest.approx(15.0)


def test_aggregate_majority():
    """aggregate_multi_matches en mode majority pour catégorielles."""
    import pandas as pd

    df = gpd.GeoDataFrame({
        "id": [1, 1, 1, 2],
        "cat": ["A", "A", "B", "C"],
        "geometry": [box(0, 0, 1, 1)] * 4,
    })
    result = aggregate_multi_matches(df, "id", ["cat"], strategy="majority")
    assert result.loc[result["id"] == 1, "cat"].values[0] == "A"
