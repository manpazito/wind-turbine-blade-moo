from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_turbine import bem
from wind_turbine.config import DesignSpaceConfig, RotorConfig
from wind_turbine.xfoil import PolarPoint


class DummyPolarsWithTables:
    def __init__(self) -> None:
        self.reynolds_bins = (200000.0, 600000.0, 1200000.0)
        self._alpha = np.arange(-8.0, 16.5, 0.5)
        self._tables: dict[float, pd.DataFrame] = {}
        for re in self.reynolds_bins:
            scale = (re / 3.0e5) ** 0.12
            cl = 0.11 * self._alpha * (0.92 + 0.06 * scale)
            cl = np.clip(cl, -1.3, 1.55)
            cd = 0.0105 / max(scale, 0.4) + 0.012 * cl**2 + 0.0002 * np.maximum(self._alpha - 9.0, 0.0) ** 2
            self._tables[re] = pd.DataFrame(
                {
                    "alpha": self._alpha,
                    "cl": cl,
                    "cd": cd,
                }
            )

    def get_design_polar(self, airfoil: str, reynolds: float) -> pd.DataFrame:
        _ = airfoil
        re = min(self.reynolds_bins, key=lambda x: abs(x - reynolds))
        return self._tables[re]

    def sample(self, airfoil: str, reynolds: float, alpha_deg: float) -> PolarPoint:
        _ = airfoil
        re_bins = self.reynolds_bins
        if reynolds <= re_bins[0]:
            lo = hi = re_bins[0]
        elif reynolds >= re_bins[-1]:
            lo = hi = re_bins[-1]
        else:
            lo, hi = re_bins[0], re_bins[-1]
            for i in range(1, len(re_bins)):
                if reynolds <= re_bins[i]:
                    lo = re_bins[i - 1]
                    hi = re_bins[i]
                    break

        def sample_row(re_val: float) -> tuple[float, float]:
            row = self._tables[re_val]
            a = row["alpha"].to_numpy()
            cl = row["cl"].to_numpy()
            cd = row["cd"].to_numpy()
            return float(np.interp(alpha_deg, a, cl)), float(np.interp(alpha_deg, a, cd))

        cl_lo, cd_lo = sample_row(lo)
        cl_hi, cd_hi = sample_row(hi)
        if hi == lo:
            return PolarPoint(cl=cl_lo, cd=cd_lo)

        w = (reynolds - lo) / max(hi - lo, 1e-12)
        return PolarPoint(
            cl=(1.0 - w) * cl_lo + w * cl_hi,
            cd=(1.0 - w) * cd_lo + w * cd_hi,
        )


def test_cpp_and_python_bem_paths_match(monkeypatch) -> None:
    if bem._evaluate_rotor_cpp_impl is None:
        pytest.skip("C++ BEM extension is not available in this environment")

    rotor = RotorConfig(radius_m=18.0, wind_speed_ms=8.2, n_sections=12)
    design_space = DesignSpaceConfig()
    polars = DummyPolarsWithTables()

    sections = bem.design_blade_geometry(
        rotor=rotor,
        design_space=design_space,
        polar_db=polars,  # type: ignore[arg-type]
        airfoil="4412",
        blades=3,
        tip_speed_ratio=7.1,
        design_aoa_deg=6.0,
        hub_radius_ratio=0.2,
        chord_scale=1.0,
        twist_scale=1.0,
    )

    cpp_sections, cpp_perf = bem.evaluate_rotor(
        rotor=rotor,
        polar_db=polars,  # type: ignore[arg-type]
        airfoil="4412",
        blades=3,
        tip_speed_ratio=7.1,
        hub_radius_ratio=0.2,
        sections=sections,
    )

    monkeypatch.setattr(bem, "_evaluate_rotor_cpp_impl", None)
    py_sections, py_perf = bem.evaluate_rotor(
        rotor=rotor,
        polar_db=polars,  # type: ignore[arg-type]
        airfoil="4412",
        blades=3,
        tip_speed_ratio=7.1,
        hub_radius_ratio=0.2,
        sections=sections,
    )

    assert len(cpp_sections) == len(py_sections)
    assert pytest.approx(py_perf.cp, rel=1e-7, abs=1e-9) == cpp_perf.cp
    assert pytest.approx(py_perf.ct, rel=1e-7, abs=1e-9) == cpp_perf.ct
    assert pytest.approx(py_perf.torque_nm, rel=1e-7, abs=1e-6) == cpp_perf.torque_nm
    assert pytest.approx(py_perf.thrust_n, rel=1e-7, abs=1e-6) == cpp_perf.thrust_n
    assert pytest.approx(py_perf.root_moment_nm, rel=1e-7, abs=1e-5) == cpp_perf.root_moment_nm
