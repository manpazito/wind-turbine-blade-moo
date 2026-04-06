from __future__ import annotations

import math

import pytest

from wind_turbine.config import XfoilConfig
from wind_turbine.xfoil import XfoilPolarDatabase


def test_surrogate_backend_generates_polars_and_coordinates(tmp_path) -> None:
    cfg = XfoilConfig(
        backend="surrogate",
        reynolds_bins=[120000, 300000, 900000],
        alpha_start_deg=-5.0,
        alpha_end_deg=14.0,
        alpha_step_deg=1.0,
    )
    db = XfoilPolarDatabase(config=cfg, cache_dir=tmp_path)
    db.prepare(["4412", "0012"])

    sample = db.sample("4412", reynolds=4.5e5, alpha_deg=6.0)
    assert math.isfinite(sample.cl)
    assert math.isfinite(sample.cd)
    assert sample.cd > 0.0

    coords = db.get_airfoil_coordinates("4412")
    assert len(coords) > 100
    assert set(coords.columns) == {"x", "y"}


def test_auto_backend_falls_back_when_executable_is_missing(tmp_path) -> None:
    cfg = XfoilConfig(
        executable="xfoil-this-command-does-not-exist",
        backend="auto",
        fallback_to_surrogate=True,
        reynolds_bins=[200000, 600000],
    )
    with pytest.warns(RuntimeWarning, match="XFOIL executable was not found"):
        db = XfoilPolarDatabase(config=cfg, cache_dir=tmp_path)
    db.prepare(["2412"])

    sample = db.sample("2412", reynolds=3.5e5, alpha_deg=4.0)
    assert sample.cd > 0.0
