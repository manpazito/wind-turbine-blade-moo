import math
import numpy as np
from turbine_blade_moo.config import DesignSpaceConfig, RotorConfig
from turbine_blade_moo.xfoil import XfoilPolarDatabase
try:
    from turbine_blade_moo._bem_cpp import evaluate_rotor_cpp as _evaluate_rotor_cpp_impl
except Exception:
    _evaluate_rotor_cpp_impl = None

class SectionGeometry:

    def __init__(self, r_m, r_over_r, chord_m, twist_deg):
        self.r_m = r_m
        self.r_over_r = r_over_r
        self.chord_m = chord_m
        self.twist_deg = twist_deg

class SectionResult:

    def __init__(self, r_m, chord_m, twist_deg, phi_deg, alpha_deg, reynolds, cl, cd, a, a_prime, cn, ct, dthrust_n, dtorque_nm, local_solidity):
        self.r_m = r_m
        self.chord_m = chord_m
        self.twist_deg = twist_deg
        self.phi_deg = phi_deg
        self.alpha_deg = alpha_deg
        self.reynolds = reynolds
        self.cl = cl
        self.cd = cd
        self.a = a
        self.a_prime = a_prime
        self.cn = cn
        self.ct = ct
        self.dthrust_n = dthrust_n
        self.dtorque_nm = dtorque_nm
        self.local_solidity = local_solidity

class RotorPerformance:

    def __init__(self, cp, ct, power_w, thrust_n, torque_nm, root_moment_nm, solidity_mean):
        self.cp = cp
        self.ct = ct
        self.power_w = power_w
        self.thrust_n = thrust_n
        self.torque_nm = torque_nm
        self.root_moment_nm = root_moment_nm
        self.solidity_mean = solidity_mean

def _clamp(value, lo, hi):
    return float(min(max(value, lo), hi))

def _prandtl_loss_factor(blades, radius_m, hub_radius_m, r_m, phi_rad):
    sin_phi = abs(math.sin(phi_rad))
    sin_phi = max(sin_phi, 1e-06)
    denom = max(r_m * sin_phi, 1e-06)
    tip_exp = -0.5 * blades * (radius_m - r_m) / denom
    root_exp = -0.5 * blades * (r_m - hub_radius_m) / denom
    tip_term = math.exp(_clamp(tip_exp, -60.0, 0.0))
    root_term = math.exp(_clamp(root_exp, -60.0, 0.0))
    tip_loss = 2.0 / math.pi * math.acos(_clamp(tip_term, 0.0, 1.0))
    root_loss = 2.0 / math.pi * math.acos(_clamp(root_term, 0.0, 1.0))
    return max(tip_loss * root_loss, 0.001)

def _build_polar_grids(polar_db, airfoil):
    re_bins_raw = getattr(polar_db, 'reynolds_bins', ())
    re_bins = np.asarray(re_bins_raw, dtype=float)
    if re_bins.ndim != 1 or re_bins.size < 1:
        return None
    try:
        polars = [polar_db.get_design_polar(airfoil, float(re)) for re in re_bins]
    except Exception:
        return None
    alpha_ref = polars[0]['alpha'].to_numpy(dtype=float)
    if alpha_ref.size < 2:
        return None
    cl_rows = []
    cd_rows = []
    for polar in polars:
        alpha = polar['alpha'].to_numpy(dtype=float)
        cl = polar['cl'].to_numpy(dtype=float)
        cd = polar['cd'].to_numpy(dtype=float)
        if alpha.size < 2:
            return None
        cl_rows.append(np.interp(alpha_ref, alpha, cl, left=cl[0], right=cl[-1]))
        cd_rows.append(np.interp(alpha_ref, alpha, cd, left=cd[0], right=cd[-1]))
    cl_grid = np.vstack(cl_rows).astype(float, copy=False)
    cd_grid = np.maximum(np.vstack(cd_rows), 1e-05).astype(float, copy=False)
    return (re_bins.astype(float, copy=False), alpha_ref.astype(float, copy=False), cl_grid, cd_grid)

def _evaluate_rotor_cpp(rotor, polar_db, airfoil, blades, tip_speed_ratio, hub_radius_ratio, sections):
    if _evaluate_rotor_cpp_impl is None or not sections:
        return None
    grids = _build_polar_grids(polar_db=polar_db, airfoil=airfoil)
    if grids is None:
        return None
    re_bins, alpha_grid, cl_grid, cd_grid = grids
    r_m = np.array([s.r_m for s in sections], dtype=float)
    chord_m = np.array([s.chord_m for s in sections], dtype=float)
    twist_deg = np.array([s.twist_deg for s in sections], dtype=float)
    try:
        out = _evaluate_rotor_cpp_impl(radius_m=float(rotor.radius_m), wind_speed_ms=float(rotor.wind_speed_ms), air_density=float(rotor.air_density), dynamic_viscosity=float(rotor.dynamic_viscosity), pitch_deg=float(rotor.pitch_deg), blades=int(blades), tip_speed_ratio=float(tip_speed_ratio), hub_radius_ratio=float(hub_radius_ratio), r_m=r_m, chord_m=chord_m, twist_deg=twist_deg, re_bins=re_bins, alpha_grid=alpha_grid, cl_grid=cl_grid, cd_grid=cd_grid)
    except Exception:
        return None

    def as_array(key):
        return np.asarray(out[key], dtype=float)
    phi_deg = as_array('phi_deg')
    if phi_deg.shape[0] != len(sections):
        return None
    alpha_deg = as_array('alpha_deg')
    reynolds = as_array('reynolds')
    cl = as_array('cl')
    cd = as_array('cd')
    a = as_array('a')
    a_prime = as_array('a_prime')
    cn = as_array('cn')
    ct = as_array('ct_section')
    dthrust_n = as_array('dthrust_n')
    dtorque_nm = as_array('dtorque_nm')
    local_solidity = as_array('local_solidity')
    section_results = []
    for i, sec in enumerate(sections):
        section_results.append(SectionResult(r_m=sec.r_m, chord_m=sec.chord_m, twist_deg=sec.twist_deg, phi_deg=float(phi_deg[i]), alpha_deg=float(alpha_deg[i]), reynolds=float(reynolds[i]), cl=float(cl[i]), cd=float(cd[i]), a=float(a[i]), a_prime=float(a_prime[i]), cn=float(cn[i]), ct=float(ct[i]), dthrust_n=float(dthrust_n[i]), dtorque_nm=float(dtorque_nm[i]), local_solidity=float(local_solidity[i])))
    perf = RotorPerformance(cp=float(out['cp']), ct=float(out['ct']), power_w=float(out['power_w']), thrust_n=float(out['thrust_n']), torque_nm=float(out['torque_nm']), root_moment_nm=float(out['root_moment_nm']), solidity_mean=float(out['solidity_mean']))
    return (section_results, perf)

def design_blade_geometry(rotor, design_space, polar_db, airfoil, blades, tip_speed_ratio, design_aoa_deg, hub_radius_ratio, chord_scale, twist_scale):
    radius_m = rotor.radius_m
    hub_radius_m = hub_radius_ratio * radius_m
    dr = (radius_m - hub_radius_m) / rotor.n_sections
    min_chord = design_space.chord_ratio_limits[0] * radius_m
    max_chord = design_space.chord_ratio_limits[1] * radius_m
    sections = []
    for i in range(rotor.n_sections):
        r_m = hub_radius_m + (i + 0.5) * dr
        lambda_r = tip_speed_ratio * (r_m / radius_m)
        phi_des = 2.0 / 3.0 * math.atan2(1.0, max(lambda_r, 1e-06))
        re_guess = 350000.0 + 2800000.0 * (r_m / radius_m)
        cl_design = max(polar_db.sample(airfoil, re_guess, design_aoa_deg).cl, 0.35)
        chord_m = chord_scale * (8.0 * math.pi * r_m * math.sin(phi_des) / (blades * cl_design * max(lambda_r, 1e-06)))
        chord_m = _clamp(chord_m, min_chord, max_chord)
        twist_base = math.degrees(phi_des) - design_aoa_deg - rotor.pitch_deg
        twist_deg = twist_base * twist_scale
        sections.append(SectionGeometry(r_m=r_m, r_over_r=r_m / radius_m, chord_m=chord_m, twist_deg=twist_deg))
    return sections

def _evaluate_rotor_python(rotor, polar_db, airfoil, blades, tip_speed_ratio, hub_radius_ratio, sections):
    radius_m = rotor.radius_m
    hub_radius_m = hub_radius_ratio * radius_m
    omega = tip_speed_ratio * rotor.wind_speed_ms / radius_m
    dr = (radius_m - hub_radius_m) / max(len(sections), 1)
    thrust_n = 0.0
    torque_nm = 0.0
    root_moment_nm = 0.0
    section_results = []
    solidity_terms = []
    for sec in sections:
        a = 0.3
        a_prime = 0.0
        last = None
        for _ in range(120):
            v_axial = rotor.wind_speed_ms * (1.0 - a)
            v_tan = omega * sec.r_m * (1.0 + a_prime)
            phi = math.atan2(max(v_axial, 1e-08), max(v_tan, 1e-08))
            w_rel = math.hypot(v_axial, v_tan)
            alpha_deg = math.degrees(phi) - (sec.twist_deg + rotor.pitch_deg)
            reynolds = rotor.air_density * w_rel * sec.chord_m / rotor.dynamic_viscosity
            polar = polar_db.sample(airfoil, reynolds, alpha_deg)
            cl = polar.cl
            cd = polar.cd
            cn = cl * math.cos(phi) + cd * math.sin(phi)
            ct = cl * math.sin(phi) - cd * math.cos(phi)
            sigma = blades * sec.chord_m / (2.0 * math.pi * sec.r_m)
            f_loss = _prandtl_loss_factor(blades, radius_m, hub_radius_m, sec.r_m, phi)
            sin_phi = max(abs(math.sin(phi)), 1e-05)
            cos_phi = max(abs(math.cos(phi)), 1e-05)
            denom_a = 4.0 * f_loss * sin_phi * sin_phi / max(sigma * cn, 1e-08)
            a_new = 1.0 / (denom_a + 1.0)
            denom_ap = 4.0 * f_loss * sin_phi * cos_phi / max(sigma * ct, 1e-08)
            a_prime_new = 1.0 / (denom_ap - 1.0)
            a_new = _clamp(a_new, 0.0, 0.95)
            a_prime_new = _clamp(a_prime_new, -0.5, 0.5)
            a_upd = 0.75 * a + 0.25 * a_new
            ap_upd = 0.75 * a_prime + 0.25 * a_prime_new
            dyn_pressure = 0.5 * rotor.air_density * w_rel ** 2
            d_lift = dyn_pressure * sec.chord_m * cl * dr
            d_drag = dyn_pressure * sec.chord_m * cd * dr
            dthrust_n = blades * (d_lift * math.cos(phi) + d_drag * math.sin(phi))
            dtorque_nm = blades * (d_lift * math.sin(phi) - d_drag * math.cos(phi)) * sec.r_m
            last = SectionResult(r_m=sec.r_m, chord_m=sec.chord_m, twist_deg=sec.twist_deg, phi_deg=math.degrees(phi), alpha_deg=alpha_deg, reynolds=float(reynolds), cl=float(cl), cd=float(cd), a=float(a_upd), a_prime=float(ap_upd), cn=float(cn), ct=float(ct), dthrust_n=float(dthrust_n), dtorque_nm=float(dtorque_nm), local_solidity=float(sigma))
            if abs(a_upd - a) < 0.0001 and abs(ap_upd - a_prime) < 0.0001:
                a = a_upd
                a_prime = ap_upd
                break
            a = a_upd
            a_prime = ap_upd
        if last is None:
            continue
        thrust_n += last.dthrust_n
        torque_nm += last.dtorque_nm
        root_moment_nm += last.dthrust_n * max(sec.r_m - hub_radius_m, 0.0)
        solidity_terms.append(last.local_solidity)
        section_results.append(last)
    swept_area = math.pi * radius_m ** 2
    power_w = omega * torque_nm
    cp = power_w / (0.5 * rotor.air_density * swept_area * rotor.wind_speed_ms ** 3 + 1e-09)
    ct = thrust_n / (0.5 * rotor.air_density * swept_area * rotor.wind_speed_ms ** 2 + 1e-09)
    solidity_mean = float(np.mean(solidity_terms)) if solidity_terms else 0.0
    perf = RotorPerformance(cp=float(cp), ct=float(ct), power_w=float(power_w), thrust_n=float(thrust_n), torque_nm=float(torque_nm), root_moment_nm=float(root_moment_nm), solidity_mean=solidity_mean)
    return (section_results, perf)

def evaluate_rotor(rotor, polar_db, airfoil, blades, tip_speed_ratio, hub_radius_ratio, sections):
    cpp_result = _evaluate_rotor_cpp(rotor=rotor, polar_db=polar_db, airfoil=airfoil, blades=blades, tip_speed_ratio=tip_speed_ratio, hub_radius_ratio=hub_radius_ratio, sections=sections)
    if cpp_result is not None:
        return cpp_result
    return _evaluate_rotor_python(rotor=rotor, polar_db=polar_db, airfoil=airfoil, blades=blades, tip_speed_ratio=tip_speed_ratio, hub_radius_ratio=hub_radius_ratio, sections=sections)
