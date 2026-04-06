import math
import numpy as np
import pandas as pd
from turbine_blade_moo.bem import RotorPerformance, design_blade_geometry, evaluate_rotor
from turbine_blade_moo.config import DesignSpaceConfig, OptimizerConfig, RotorConfig
from turbine_blade_moo.xfoil import XfoilPolarDatabase

class DesignVector:

    def __init__(self, airfoil_idx, blades, tip_speed_ratio, aoa_deg, hub_radius_ratio, chord_scale, twist_scale):
        self.airfoil_idx = airfoil_idx
        self.blades = blades
        self.tip_speed_ratio = tip_speed_ratio
        self.aoa_deg = aoa_deg
        self.hub_radius_ratio = hub_radius_ratio
        self.chord_scale = chord_scale
        self.twist_scale = twist_scale

    def cache_key(self):
        return (self.airfoil_idx, self.blades, round(self.tip_speed_ratio, 6), round(self.aoa_deg, 6), round(self.hub_radius_ratio, 6), round(self.chord_scale, 6), round(self.twist_scale, 6))

    def with_updates(self, **kwargs):
        return DesignVector(airfoil_idx=kwargs.get('airfoil_idx', self.airfoil_idx), blades=kwargs.get('blades', self.blades), tip_speed_ratio=kwargs.get('tip_speed_ratio', self.tip_speed_ratio), aoa_deg=kwargs.get('aoa_deg', self.aoa_deg), hub_radius_ratio=kwargs.get('hub_radius_ratio', self.hub_radius_ratio), chord_scale=kwargs.get('chord_scale', self.chord_scale), twist_scale=kwargs.get('twist_scale', self.twist_scale))

class EvaluatedDesign:

    def __init__(self, design, airfoil, sections, section_results, performance):
        self.design = design
        self.airfoil = airfoil
        self.sections = sections
        self.section_results = section_results
        self.performance = performance

    @property
    def objectives(self):
        return (-self.performance.cp, self.performance.root_moment_nm, self.performance.solidity_mean)

    def to_dict(self):
        return {'airfoil': self.airfoil, 'blades': self.design.blades, 'tip_speed_ratio': self.design.tip_speed_ratio, 'aoa_deg': self.design.aoa_deg, 'hub_radius_ratio': self.design.hub_radius_ratio, 'chord_scale': self.design.chord_scale, 'twist_scale': self.design.twist_scale, 'cp': self.performance.cp, 'ct': self.performance.ct, 'power_w': self.performance.power_w, 'thrust_n': self.performance.thrust_n, 'torque_nm': self.performance.torque_nm, 'root_moment_nm': self.performance.root_moment_nm, 'solidity_mean': self.performance.solidity_mean, 'obj_neg_cp': -self.performance.cp, 'obj_root_moment': self.performance.root_moment_nm, 'obj_solidity': self.performance.solidity_mean}

class OptimizationOutcome:

    def __init__(self, final_population, all_evaluations, pareto_front, best_compromise):
        self.final_population = final_population
        self.all_evaluations = all_evaluations
        self.pareto_front = pareto_front
        self.best_compromise = best_compromise

def _dominates(lhs, rhs):
    lhs_arr = np.array(lhs)
    rhs_arr = np.array(rhs)
    return bool(np.all(lhs_arr <= rhs_arr) and np.any(lhs_arr < rhs_arr))

def fast_non_dominated_sort(items):
    n = len(items)
    dominates_list = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    rank = np.full(n, -1, dtype=int)
    fronts = [[]]
    objectives = [it.objectives for it in items]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return (fronts, rank)

def crowding_distance(items, front):
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float('inf') for idx in front}
    distances = {idx: 0.0 for idx in front}
    obj_array = np.array([items[idx].objectives for idx in front], dtype=float)
    for j in range(obj_array.shape[1]):
        order = np.argsort(obj_array[:, j])
        f_sorted = [front[k] for k in order]
        distances[f_sorted[0]] = float('inf')
        distances[f_sorted[-1]] = float('inf')
        min_v = obj_array[order[0], j]
        max_v = obj_array[order[-1], j]
        scale = max(max_v - min_v, 1e-12)
        for k in range(1, len(order) - 1):
            if math.isinf(distances[f_sorted[k]]):
                continue
            prev_v = obj_array[order[k - 1], j]
            next_v = obj_array[order[k + 1], j]
            distances[f_sorted[k]] += float((next_v - prev_v) / scale)
    return distances

def _range_sample(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))

def random_design(rng, design_space):
    return DesignVector(airfoil_idx=int(rng.integers(0, len(design_space.airfoils))), blades=int(rng.choice(np.array(design_space.blades_options))), tip_speed_ratio=_range_sample(rng, design_space.tip_speed_ratio_range), aoa_deg=_range_sample(rng, design_space.aoa_deg_range), hub_radius_ratio=_range_sample(rng, design_space.hub_radius_ratio_range), chord_scale=_range_sample(rng, design_space.chord_scale_range), twist_scale=_range_sample(rng, design_space.twist_scale_range))

def _clip_to_bounds(design, design_space):
    return DesignVector(airfoil_idx=int(max(0, min(design.airfoil_idx, len(design_space.airfoils) - 1))), blades=int(min(max(design.blades, min(design_space.blades_options)), max(design_space.blades_options))), tip_speed_ratio=float(min(max(design.tip_speed_ratio, design_space.tip_speed_ratio_range[0]), design_space.tip_speed_ratio_range[1])), aoa_deg=float(min(max(design.aoa_deg, design_space.aoa_deg_range[0]), design_space.aoa_deg_range[1])), hub_radius_ratio=float(min(max(design.hub_radius_ratio, design_space.hub_radius_ratio_range[0]), design_space.hub_radius_ratio_range[1])), chord_scale=float(min(max(design.chord_scale, design_space.chord_scale_range[0]), design_space.chord_scale_range[1])), twist_scale=float(min(max(design.twist_scale, design_space.twist_scale_range[0]), design_space.twist_scale_range[1])))

def crossover_and_mutate(p1, p2, rng, design_space, cfg):
    if rng.random() < cfg.crossover_probability:
        beta = float(rng.uniform(0.0, 1.0))
        child = DesignVector(airfoil_idx=int(p1.airfoil_idx if rng.random() < 0.5 else p2.airfoil_idx), blades=int(p1.blades if rng.random() < 0.5 else p2.blades), tip_speed_ratio=p1.tip_speed_ratio + beta * (p2.tip_speed_ratio - p1.tip_speed_ratio), aoa_deg=p1.aoa_deg + beta * (p2.aoa_deg - p1.aoa_deg), hub_radius_ratio=p1.hub_radius_ratio + beta * (p2.hub_radius_ratio - p1.hub_radius_ratio), chord_scale=p1.chord_scale + beta * (p2.chord_scale - p1.chord_scale), twist_scale=p1.twist_scale + beta * (p2.twist_scale - p1.twist_scale))
    else:
        child = p1
    if rng.random() < cfg.mutation_probability:
        child = DesignVector(airfoil_idx=int(rng.integers(0, len(design_space.airfoils))), blades=int(rng.choice(np.array(design_space.blades_options))), tip_speed_ratio=child.tip_speed_ratio, aoa_deg=child.aoa_deg, hub_radius_ratio=child.hub_radius_ratio, chord_scale=child.chord_scale, twist_scale=child.twist_scale)
    if rng.random() < cfg.mutation_probability:
        tsr_span = design_space.tip_speed_ratio_range[1] - design_space.tip_speed_ratio_range[0]
        child = DesignVector(airfoil_idx=child.airfoil_idx, blades=child.blades, tip_speed_ratio=child.tip_speed_ratio + float(rng.normal(0.0, 0.08 * tsr_span)), aoa_deg=child.aoa_deg, hub_radius_ratio=child.hub_radius_ratio, chord_scale=child.chord_scale, twist_scale=child.twist_scale)
    if rng.random() < cfg.mutation_probability:
        aoa_span = design_space.aoa_deg_range[1] - design_space.aoa_deg_range[0]
        child = DesignVector(airfoil_idx=child.airfoil_idx, blades=child.blades, tip_speed_ratio=child.tip_speed_ratio, aoa_deg=child.aoa_deg + float(rng.normal(0.0, 0.08 * aoa_span)), hub_radius_ratio=child.hub_radius_ratio, chord_scale=child.chord_scale, twist_scale=child.twist_scale)
    if rng.random() < cfg.mutation_probability:
        hub_span = design_space.hub_radius_ratio_range[1] - design_space.hub_radius_ratio_range[0]
        child = DesignVector(airfoil_idx=child.airfoil_idx, blades=child.blades, tip_speed_ratio=child.tip_speed_ratio, aoa_deg=child.aoa_deg, hub_radius_ratio=child.hub_radius_ratio + float(rng.normal(0.0, 0.08 * hub_span)), chord_scale=child.chord_scale, twist_scale=child.twist_scale)
    if rng.random() < cfg.mutation_probability:
        c_span = design_space.chord_scale_range[1] - design_space.chord_scale_range[0]
        child = DesignVector(airfoil_idx=child.airfoil_idx, blades=child.blades, tip_speed_ratio=child.tip_speed_ratio, aoa_deg=child.aoa_deg, hub_radius_ratio=child.hub_radius_ratio, chord_scale=child.chord_scale + float(rng.normal(0.0, 0.08 * c_span)), twist_scale=child.twist_scale)
    if rng.random() < cfg.mutation_probability:
        t_span = design_space.twist_scale_range[1] - design_space.twist_scale_range[0]
        child = DesignVector(airfoil_idx=child.airfoil_idx, blades=child.blades, tip_speed_ratio=child.tip_speed_ratio, aoa_deg=child.aoa_deg, hub_radius_ratio=child.hub_radius_ratio, chord_scale=child.chord_scale, twist_scale=child.twist_scale + float(rng.normal(0.0, 0.08 * t_span)))
    return _clip_to_bounds(child, design_space)

def _tournament_pick(population, rank, crowding, rng):
    i, j = (int(rng.integers(0, len(population))), int(rng.integers(0, len(population))))
    if rank[i] < rank[j]:
        return population[i]
    if rank[j] < rank[i]:
        return population[j]
    if crowding.get(i, 0.0) > crowding.get(j, 0.0):
        return population[i]
    return population[j]

def _environmental_selection(candidates, population_size):
    fronts, _ = fast_non_dominated_sort(candidates)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend((candidates[idx] for idx in front))
            continue
        distances = crowding_distance(candidates, front)
        ordered = sorted(front, key=lambda idx: distances.get(idx, 0.0), reverse=True)
        remaining = population_size - len(selected)
        selected.extend((candidates[idx] for idx in ordered[:remaining]))
        break
    return selected

def evaluate_design(design, rotor, design_space, polar_db):
    airfoil = design_space.airfoils[design.airfoil_idx]
    sections = design_blade_geometry(rotor=rotor, design_space=design_space, polar_db=polar_db, airfoil=airfoil, blades=design.blades, tip_speed_ratio=design.tip_speed_ratio, design_aoa_deg=design.aoa_deg, hub_radius_ratio=design.hub_radius_ratio, chord_scale=design.chord_scale, twist_scale=design.twist_scale)
    section_results, perf = evaluate_rotor(rotor=rotor, polar_db=polar_db, airfoil=airfoil, blades=design.blades, tip_speed_ratio=design.tip_speed_ratio, hub_radius_ratio=design.hub_radius_ratio, sections=sections)
    if not math.isfinite(perf.cp):
        perf = RotorPerformance(cp=-1.0, ct=0.0, power_w=0.0, thrust_n=0.0, torque_nm=0.0, root_moment_nm=1000000000000000.0, solidity_mean=1000000.0)
    return EvaluatedDesign(design=design, airfoil=airfoil, sections=sections, section_results=section_results, performance=perf)

def run_nsga2(rotor, design_space, optimizer_cfg, polar_db, seed):
    rng = np.random.default_rng(seed)
    eval_cache = {}

    def get_eval(design):
        key = design.cache_key()
        if key not in eval_cache:
            eval_cache[key] = evaluate_design(design, rotor, design_space, polar_db)
        return eval_cache[key]
    population = [random_design(rng, design_space) for _ in range(optimizer_cfg.population_size)]
    evaluated_population = [get_eval(p) for p in population]
    all_evaluations = list(evaluated_population)
    for _ in range(optimizer_cfg.generations):
        fronts, rank = fast_non_dominated_sort(evaluated_population)
        crowding = {}
        for front in fronts:
            crowding.update(crowding_distance(evaluated_population, front))
        offspring_designs = []
        while len(offspring_designs) < optimizer_cfg.population_size:
            p1 = _tournament_pick(evaluated_population, rank, crowding, rng)
            p2 = _tournament_pick(evaluated_population, rank, crowding, rng)
            child = crossover_and_mutate(p1=p1.design, p2=p2.design, rng=rng, design_space=design_space, cfg=optimizer_cfg)
            offspring_designs.append(child)
        offspring_eval = [get_eval(x) for x in offspring_designs]
        all_evaluations.extend(offspring_eval)
        combined = evaluated_population + offspring_eval
        evaluated_population = _environmental_selection(combined, optimizer_cfg.population_size)
    fronts, _ = fast_non_dominated_sort(evaluated_population)
    pareto = [evaluated_population[idx] for idx in fronts[0]] if fronts and fronts[0] else evaluated_population
    pareto = _deduplicate_designs(pareto)
    best = choose_best_compromise(pareto)
    return OptimizationOutcome(final_population=evaluated_population, all_evaluations=_deduplicate_designs(all_evaluations), pareto_front=pareto, best_compromise=best)

def _deduplicate_designs(items):
    uniq = {}
    for item in items:
        uniq[item.design.cache_key()] = item
    return list(uniq.values())

def choose_best_compromise(pareto):
    if not pareto:
        raise ValueError('Pareto set is empty.')
    objs = np.array([p.objectives for p in pareto], dtype=float)
    ideal = objs.min(axis=0)
    nadir = objs.max(axis=0)
    norm = (objs - ideal) / (nadir - ideal + 1e-12)
    dist = np.linalg.norm(norm, axis=1)
    return pareto[int(np.argmin(dist))]

def to_dataframe(items):
    return pd.DataFrame([x.to_dict() for x in items]).sort_values('cp', ascending=False).reset_index(drop=True)
