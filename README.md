# Turbine Blade Multi-Objective Optimization

*Li et al. (2023)-inspired computational framework for aerodynamic design*

This repository presents a wind turbine blade optimization framework grounded in the formulation of blade design as a dynamic, multi-objective problem. Rather than seeking a single optimum, the system constructs Pareto-optimal solutions across competing aerodynamic and structural criteria.

The implementation integrates airfoil polar evaluation (XFOIL or surrogate-based), rotor-scale performance modeling via Blade Element Momentum (BEM) theory, and population-based multi-objective optimization (NSGA-II style). The resulting pipeline enables systematic exploration of the coupled geometry–performance design space.

---

## Problem Formulation

Blade design is expressed as a structured decision vector spanning airfoil selection, blade count, tip-speed ratio, angle of attack, hub ratio, and chord/twist scaling.

System performance is evaluated through competing objectives, including power coefficient (`Cp`), root bending moment, and rotor solidity. Optimization proceeds by iteratively sampling this space and extracting the Pareto front, thereby exposing the trade structure governing feasible designs.

---

## Methodological Positioning

This work follows the conceptual framework of Li et al. (2023), which characterizes turbine blade optimization as multi-objective, sequential in its treatment of design variables, and constrained by the cost of aerodynamic evaluation.

The present implementation preserves this structure while introducing practical substitutions: evolutionary search in place of reinforcement learning, BEM in place of full CFD cascade evaluation, and analytic surrogate polars in place of learned CFD surrogates. These choices prioritize reproducibility and computational efficiency while maintaining the underlying optimization logic.

---

## Computational Structure

The pipeline consists of four coupled components:

1. Aerodynamic modeling via cached XFOIL or surrogate polars
2. Rotor analysis using iterative BEM evaluation
3. Multi-objective optimization via non-dominated sorting and crowding distance
4. Post-processing including Pareto extraction, compromise selection, and sensitivity analysis

---

## Usage

The framework is exposed both as a command-line workflow and a Python API.

**CLI execution**

```bash
python -m pip install -e .
python -m wind_turbine_blade_moo --config configs/quick_test.yaml
wtb-moo --config configs/quick_test.yaml
```

**Python API**

```python
from wind_turbine_blade_moo.config import load_config
from wind_turbine_blade_moo.pipeline import run_pipeline

cfg = load_config("configs/quick_test.yaml")
summary = run_pipeline(cfg)

print(summary["cp"], summary["aero_backend"])
```

Execution produces a full design population, the Pareto-optimal subset, and a selected compromise solution with associated geometry and performance metrics.

---

## Interpretation

This repository should be read as a computational instantiation of a broader methodological claim: that wind turbine blade design is fundamentally a Pareto-structured, dynamically reconfigurable problem.

Accordingly, the system emphasizes explicit tradeoff representation, rapid re-optimization under varying constraints, and separation between model fidelity and optimization logic.

---

## References

[1] Lele Li, Weihao Zhang, Ya Li, Chiju Jiang, and Yufan Wang, "Multi-objective optimization of turbine blade profiles based on multi-agent reinforcement learning," *Energy Conversion and Management* 297 (2023): 117637. https://doi.org/10.1016/j.enconman.2023.117637
