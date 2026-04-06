from pathlib import Path
import yaml

class ProjectConfig:

    def __init__(self, output_dir=Path('outputs'), random_seed=42):
        self.output_dir = output_dir
        self.random_seed = random_seed

class RotorConfig:

    def __init__(self, radius_m=40.0, wind_speed_ms=9.0, air_density=1.225, dynamic_viscosity=1.81e-05, pitch_deg=0.0, n_sections=18):
        self.radius_m = radius_m
        self.wind_speed_ms = wind_speed_ms
        self.air_density = air_density
        self.dynamic_viscosity = dynamic_viscosity
        self.pitch_deg = pitch_deg
        self.n_sections = n_sections

class DesignSpaceConfig:

    def __init__(self, airfoils=None, blades_options=None, tip_speed_ratio_range=(4.5, 10.0), aoa_deg_range=(3.0, 11.0), hub_radius_ratio_range=(0.15, 0.28), chord_scale_range=(0.75, 1.35), twist_scale_range=(0.85, 1.2), chord_ratio_limits=(0.015, 0.12)):
        self.airfoils = airfoils or ['4412', '4415', '2412', '0012', '23012']
        self.blades_options = blades_options or [2, 3, 4, 5]
        self.tip_speed_ratio_range = tip_speed_ratio_range
        self.aoa_deg_range = aoa_deg_range
        self.hub_radius_ratio_range = hub_radius_ratio_range
        self.chord_scale_range = chord_scale_range
        self.twist_scale_range = twist_scale_range
        self.chord_ratio_limits = chord_ratio_limits

class OptimizerConfig:

    def __init__(self, population_size=48, generations=18, crossover_probability=0.9, mutation_probability=0.25):
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability

class XfoilConfig:

    def __init__(self, executable='xfoil', backend='auto', fallback_to_surrogate=True, reynolds_bins=None, alpha_start_deg=-6.0, alpha_end_deg=16.0, alpha_step_deg=0.5, ncrit=9.0, max_iter=150, timeout_s=25.0):
        self.executable = executable
        self.backend = backend
        self.fallback_to_surrogate = fallback_to_surrogate
        self.reynolds_bins = reynolds_bins or [150000, 300000, 600000, 1000000, 1800000, 3000000]
        self.alpha_start_deg = alpha_start_deg
        self.alpha_end_deg = alpha_end_deg
        self.alpha_step_deg = alpha_step_deg
        self.ncrit = ncrit
        self.max_iter = max_iter
        self.timeout_s = timeout_s

class Config:

    def __init__(self, project=None, rotor=None, design_space=None, optimizer=None, xfoil=None):
        self.project = project or ProjectConfig()
        self.rotor = rotor or RotorConfig()
        self.design_space = design_space or DesignSpaceConfig()
        self.optimizer = optimizer or OptimizerConfig()
        self.xfoil = xfoil or XfoilConfig()

def _to_tuple2(raw, name):
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f'`{name}` must be a list of length 2.')
    lo = float(raw[0])
    hi = float(raw[1])
    if lo >= hi:
        raise ValueError(f'`{name}` lower bound must be less than upper bound.')
    return (lo, hi)

def _merge_config(config_cls, raw):
    if raw is None:
        return config_cls()
    defaults = config_cls()
    kwargs = {}
    for key in vars(defaults):
        if key in raw:
            kwargs[key] = raw[key]
    return config_cls(**kwargs)

def load_config(path):
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    project_raw = data.get('project', {})
    rotor_raw = data.get('rotor', {})
    design_raw = data.get('design_space', {})
    optimizer_raw = data.get('optimizer', {})
    xfoil_raw = data.get('xfoil', {})
    project = _merge_config(ProjectConfig, project_raw)
    rotor = _merge_config(RotorConfig, rotor_raw)
    optimizer = _merge_config(OptimizerConfig, optimizer_raw)
    xfoil = _merge_config(XfoilConfig, xfoil_raw)
    design_space = _merge_config(DesignSpaceConfig, design_raw)
    design_space = DesignSpaceConfig(airfoils=[str(a).strip() for a in design_space.airfoils], blades_options=[int(b) for b in design_space.blades_options], tip_speed_ratio_range=_to_tuple2(design_space.tip_speed_ratio_range, 'tip_speed_ratio_range'), aoa_deg_range=_to_tuple2(design_space.aoa_deg_range, 'aoa_deg_range'), hub_radius_ratio_range=_to_tuple2(design_space.hub_radius_ratio_range, 'hub_radius_ratio_range'), chord_scale_range=_to_tuple2(design_space.chord_scale_range, 'chord_scale_range'), twist_scale_range=_to_tuple2(design_space.twist_scale_range, 'twist_scale_range'), chord_ratio_limits=_to_tuple2(design_space.chord_ratio_limits, 'chord_ratio_limits'))
    if not design_space.airfoils:
        raise ValueError('At least one airfoil must be provided.')
    if not design_space.blades_options:
        raise ValueError('At least one blade count option must be provided.')
    if any((b < 1 for b in design_space.blades_options)):
        raise ValueError('All blade counts must be >= 1.')
    if rotor.n_sections < 4:
        raise ValueError('`rotor.n_sections` must be >= 4.')
    if rotor.radius_m <= 0.0:
        raise ValueError('`rotor.radius_m` must be > 0.')
    if optimizer.population_size < 8:
        raise ValueError('`optimizer.population_size` must be >= 8.')
    if optimizer.generations < 1:
        raise ValueError('`optimizer.generations` must be >= 1.')
    backend = str(xfoil.backend).strip().lower()
    if backend not in {'auto', 'xfoil', 'surrogate'}:
        raise ValueError('`xfoil.backend` must be one of: auto, xfoil, surrogate.')
    if not xfoil.reynolds_bins:
        raise ValueError('`xfoil.reynolds_bins` must not be empty.')
    if any((float(v) <= 0.0 for v in xfoil.reynolds_bins)):
        raise ValueError('`xfoil.reynolds_bins` values must all be > 0.')
    if xfoil.alpha_step_deg <= 0:
        raise ValueError('`xfoil.alpha_step_deg` must be > 0.')
    if xfoil.alpha_end_deg <= xfoil.alpha_start_deg:
        raise ValueError('`xfoil.alpha_end_deg` must be greater than `xfoil.alpha_start_deg`.')
    out_dir = Path(project.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    project = ProjectConfig(output_dir=out_dir, random_seed=int(project.random_seed))
    xfoil = XfoilConfig(executable=str(xfoil.executable), backend=backend, fallback_to_surrogate=bool(xfoil.fallback_to_surrogate), reynolds_bins=sorted((float(v) for v in xfoil.reynolds_bins)), alpha_start_deg=float(xfoil.alpha_start_deg), alpha_end_deg=float(xfoil.alpha_end_deg), alpha_step_deg=float(xfoil.alpha_step_deg), ncrit=float(xfoil.ncrit), max_iter=int(xfoil.max_iter), timeout_s=float(xfoil.timeout_s))
    return Config(project=project, rotor=rotor, design_space=design_space, optimizer=optimizer, xfoil=xfoil)
