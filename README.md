# Wind Turbine Blade Optimization

A from-scratch wind-turbine blade optimization workflow that combines:

- 2D airfoil polar modeling (XFOIL or built-in surrogate backend)
- 3D rotor performance prediction (BEM)
- Multi-objective search (NSGA-II style)

It optimizes blade profile, angle of attack, chord/twist scaling, and blade count while tracking aerodynamic tradeoffs such as `Cp`, root bending moment, and solidity.

## What makes this repo GitHub-ready

- Portable quick run that works even if `xfoil` is not installed (surrogate backend)
- High-fidelity XFOIL mode still available when `xfoil` is installed
- C++-accelerated BEM solver path (pybind11) with automatic Python fallback
- CLI entrypoint + YAML configs
- Automated tests
- GitHub Actions CI (`pytest` + smoke run)
- Example notebook demo: `notebooks/demo.ipynb`

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For development + notebook tools:

```bash
python -m pip install -e .[dev]
```

## Python Package Usage

Once installed, use as a package:

```python
from wind_turbine.config import load_config
from wind_turbine.pipeline import run_pipeline

cfg = load_config("configs/quick_test.yaml")
summary = run_pipeline(cfg)
print(summary["cp"], summary["aero_backend"])
```

Use the CLI entrypoint:

```bash
wt-opt --config configs/quick_test.yaml
```

## C++ Acceleration

- The package includes an optional C++ extension: `wind_turbine._bem_cpp`
- During install, the build tries to compile it automatically.
- If compilation is not available on a machine, the package still works via the pure-Python BEM fallback.
- Runtime selection is automatic; no config switch is required.

## Run

Portable quick demo (works on any machine, no external XFOIL dependency required):

```bash
python -m wind_turbine --config configs/quick_test.yaml
```

Default run (auto backend):

```bash
python -m wind_turbine --config configs/default.yaml
```

High-accuracy run (requires system `xfoil` installed):

```bash
python -m wind_turbine --config configs/high_accuracy.yaml
```

You can also use the installed script: `wt-opt --config configs/quick_test.yaml`

## Notebook Demo

Open and run:

- `notebooks/demo.ipynb`

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook runs a compact optimization and then visualizes/inspects the generated results.

## Outputs

Each run writes into the configured output directory (for example `outputs_quick/` or `outputs_notebook_demo/`):

- `all_designs.csv`: all evaluated designs
- `pareto.csv`: Pareto-optimal designs
- `best_sections.csv`: radial section geometry + aerodynamic states
- `best_airfoil_coords.csv`: selected airfoil profile coordinates
- `local_sensitivity.csv`: local perturbation sensitivity around selected design
- `best_geometry.png`: chord and twist distributions
- `best_airfoil_profile.png`: selected airfoil shape plot
- `pareto_cp_vs_moment.png`: Pareto scatter (`Cp` vs root moment)
- `summary.json`: selected compromise design summary
- `report.md`: equation-based design justification

## Backend Modes (`xfoil.backend`)

In YAML config:

- `surrogate`: always use built-in surrogate polar model (fast, portable)
- `xfoil`: always use system XFOIL executable (`xfoil.executable`)
- `auto`: use XFOIL if available, else fall back to surrogate

`xfoil.fallback_to_surrogate` controls whether runtime XFOIL failures should automatically degrade to surrogate mode.

## Equations implemented

- `lambda = Omega R / V_inf`
- `phi = atan((1-a)/(lambda_r(1+a')))`
- `alpha = phi - (theta + beta_pitch)`
- `C_n = C_l cos(phi) + C_d sin(phi)`
- `C_t = C_l sin(phi) - C_d cos(phi)`
- `a = 1 / ((4F sin^2(phi))/(sigma C_n) + 1)`
- `a' = 1 / ((4F sin(phi)cos(phi))/(sigma C_t) - 1)`
- `dT = 0.5 rho W^2 B c C_n dr`
- `dQ = 0.5 rho W^2 B c C_t r dr`
- `Cp = P / (0.5 rho A V_inf^3), P = Omega * integral(dQ)`
- `theta(r) = phi_des(r) - alpha_design - beta_pitch`
- `c(r) ~ 8 pi r sin(phi_des)/(B Cl_des lambda_r)`

## Tests

```bash
pytest -q
```

## Citation

This project is inspired by:

Lele Li, Weihao Zhang, Ya Li, Chiju Jiang, and Yufan Wang,
“Multi-Objective Optimization of Turbine Blade Profiles Based on Multi-Agent Reinforcement Learning,”
*Energy Conversion and Management* 297 (2023): 117637,
https://doi.org/10.1016/j.enconman.2023.117637.
