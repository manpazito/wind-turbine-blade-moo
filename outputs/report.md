# Wind Turbine Optimization Report

This run follows the multi-objective spirit of the provided paper, but uses a wind-turbine workflow:
XFOIL polars + BEM simulation + NSGA-II search on blade profile, AoA, twist scaling, TSR, and blade count.
The third objective includes an explicit blade-count preference penalty: target B=3, weight=0.0400.

## Core Equations Used

1. Tip speed ratio:
   - `lambda = Omega * R / V_inf`
2. Inflow angle and angle of attack:
   - `phi = atan((1-a) / (lambda_r * (1+a')))`
   - `alpha = phi - (theta + beta_pitch)`
3. Force coefficients:
   - `C_n = C_l cos(phi) + C_d sin(phi)`
   - `C_t = C_l sin(phi) - C_d cos(phi)`
4. Induction factors (BEM form):
   - `a = 1 / ( (4F sin^2(phi))/(sigma C_n) + 1 )`
   - `a' = 1 / ( (4F sin(phi)cos(phi))/(sigma C_t) - 1 )`
5. Section loads and rotor power:
   - `dT = 0.5 rho W^2 B c C_n dr`
   - `dQ = 0.5 rho W^2 B c C_t r dr`
   - `Cp = P / (0.5 rho A V_inf^3),  P = Omega * integral(dQ)`
6. Geometry synthesis equations:
   - `phi_des(r) ~= atan((1-a_des)/lambda_r), a_des~=1/3`
   - `theta(r) = phi_des(r) - alpha_design - beta_pitch`
   - `sigma_des = 4 F sin^2(phi_des) a_des / ((1-a_des) C_n,des), c(r)=2 pi r sigma_des/B`

## Optimized Design (Best Compromise From Pareto)

- Airfoil profile: `NACA 4415`
- Blade number: `3`
- Design AoA: `8.522 deg`
- Tip speed ratio: `6.557`
- Hub radius ratio: `0.2627`
- Twist scale: `0.9473`
- Chord scale: `0.7939`
- Achieved Cp: `0.4326`
- Root moment: `2666157.74 N.m`
- Mean solidity: `0.03657`

## Justification

- Blade profile choice (`NACA 4415`): selected on the Pareto front where it balances high Cp and moderate root load.
- Angle of attack: optimized AoA `8.522 deg`; XFOIL at Re≈4017581 gives peak Cl/Cd around `5.500 deg` (Cl/Cd≈162.760), so chosen AoA is in a high-efficiency aerodynamic band.
- Twist: optimized from `theta(r)=phi_des-alpha_design-beta_pitch`; resulting twist decreases from `10.637 deg` at root to `-2.459 deg` near tip, which is physically consistent with lower inflow angle at larger radius.
- Blade number: `3` emerged from multi-objective trade-off with explicit preference target `B=3`.

## Pareto Mean Performance by Blade Count

- B=2: Cp=0.3042, RootMoment=1797630.63 N.m, Solidity=0.05914
- B=3: Cp=0.3995, RootMoment=2518188.52 N.m, Solidity=0.07028
- B=4: Cp=0.4804, RootMoment=3390644.28 N.m, Solidity=0.07933
- B=5: Cp=0.4937, RootMoment=3610007.03 N.m, Solidity=0.07103

## Pareto Mean Performance by Airfoil

- NACA 4415: Cp=0.4756, RootMoment=3416421.09 N.m, Solidity=0.06090
- NACA 4412: Cp=0.4597, RootMoment=3126191.54 N.m, Solidity=0.06012
- NACA 2412: Cp=0.4143, RootMoment=2707852.65 N.m, Solidity=0.08066
- NACA 23012: Cp=0.3920, RootMoment=2484420.23 N.m, Solidity=0.06659
- NACA 0012: Cp=0.3145, RootMoment=1861838.48 N.m, Solidity=0.07773

## Notes

- This is an aerodynamic optimization model. Structural and fatigue constraints should be added before manufacturing decisions.
- The workflow is deterministic for a fixed random seed and config file.
