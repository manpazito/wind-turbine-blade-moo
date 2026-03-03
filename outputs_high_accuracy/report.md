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

- Airfoil profile: `NACA 23012`
- Blade number: `3`
- Design AoA: `8.309 deg`
- Tip speed ratio: `7.111`
- Hub radius ratio: `0.3000`
- Twist scale: `0.9942`
- Chord scale: `0.7267`
- Achieved Cp: `0.4126`
- Root moment: `2310426.76 N.m`
- Mean solidity: `0.03212`

## Justification

- Blade profile choice (`NACA 23012`): selected on the Pareto front where it balances high Cp and moderate root load.
- Angle of attack: optimized AoA `8.309 deg`; XFOIL at Re≈4285622 gives peak Cl/Cd around `8.750 deg` (Cl/Cd≈141.382), so chosen AoA is in a high-efficiency aerodynamic band.
- Twist: optimized from `theta(r)=phi_des-alpha_design-beta_pitch`; resulting twist decreases from `8.420 deg` at root to `-2.877 deg` near tip, which is physically consistent with lower inflow angle at larger radius.
- Blade number: `3` emerged from multi-objective trade-off with explicit preference target `B=3`.

## Pareto Mean Performance by Blade Count

- B=2: Cp=0.2861, RootMoment=1557066.87 N.m, Solidity=0.05538
- B=3: Cp=0.3658, RootMoment=2266381.95 N.m, Solidity=0.06864
- B=4: Cp=0.4694, RootMoment=3225587.57 N.m, Solidity=0.06260
- B=5: Cp=0.4838, RootMoment=3439687.22 N.m, Solidity=0.06994
- B=6: Cp=0.4763, RootMoment=3338170.55 N.m, Solidity=0.07045

## Pareto Mean Performance by Airfoil

- NACA 4412: Cp=0.4904, RootMoment=3673647.50 N.m, Solidity=0.07438
- NACA 6409: Cp=0.4576, RootMoment=3191503.69 N.m, Solidity=0.06667
- NACA 25012: Cp=0.4415, RootMoment=2672189.47 N.m, Solidity=0.06150
- NACA 4415: Cp=0.4367, RootMoment=2937214.41 N.m, Solidity=0.04753
- NACA 23012: Cp=0.3852, RootMoment=2210126.08 N.m, Solidity=0.05411
- NACA 24012: Cp=0.3584, RootMoment=2084599.30 N.m, Solidity=0.07818
- NACA 0012: Cp=0.2521, RootMoment=1367603.01 N.m, Solidity=0.08506

## Notes

- This is an aerodynamic optimization model. Structural and fatigue constraints should be added before manufacturing decisions.
- The workflow is deterministic for a fixed random seed and config file.
