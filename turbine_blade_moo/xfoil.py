import math
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from turbine_blade_moo.config import XfoilConfig

class PolarPoint:

    def __init__(self, cl, cd):
        self.cl = cl
        self.cd = cd

def _looks_like_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False

def _sanitize_airfoil_name(name):
    cleaned = re.sub('[^a-zA-Z0-9]+', '_', name.strip())
    return cleaned.strip('_').lower() or 'airfoil'

def _airfoil_digits(airfoil):
    return ''.join((ch for ch in airfoil if ch.isdigit()))

def _airfoil_shape_params(airfoil):
    digits = _airfoil_digits(airfoil)
    if len(digits) >= 4:
        t = max(0.06, min(int(digits[-2:]) / 100.0, 0.24))
        m = max(0.0, min(int(digits[0]) / 100.0, 0.09))
        p_raw = int(digits[1])
        p = p_raw / 10.0 if p_raw > 0 else 0.4
        p = max(0.1, min(p, 0.9))
        return (m, p, t)
    if len(digits) >= 2:
        t = max(0.06, min(int(digits[-2:]) / 100.0, 0.24))
        return (0.02, 0.4, t)
    return (0.02, 0.4, 0.12)

class XfoilPolarDatabase:

    def __init__(self, config, cache_dir):
        self.cfg = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._polar_cache = {}
        self._coords_cache = {}
        self._sorted_re = sorted((float(v) for v in self.cfg.reynolds_bins))
        self._backend = self._resolve_backend()

    @property
    def backend(self):
        return self._backend

    @property
    def reynolds_bins(self):
        return tuple(self._sorted_re)

    def prepare(self, airfoils):
        for af in airfoils:
            for re_bin in self._sorted_re:
                self._load_or_generate(af, re_bin)

    def get_design_polar(self, airfoil, reynolds):
        re_bin = self._closest_re_bin(reynolds)
        return self._load_or_generate(airfoil, re_bin)

    def sample(self, airfoil, reynolds, alpha_deg):
        if reynolds <= self._sorted_re[0]:
            return self._sample_single(airfoil, self._sorted_re[0], alpha_deg)
        if reynolds >= self._sorted_re[-1]:
            return self._sample_single(airfoil, self._sorted_re[-1], alpha_deg)
        lo = self._sorted_re[0]
        hi = self._sorted_re[-1]
        for idx in range(1, len(self._sorted_re)):
            if reynolds <= self._sorted_re[idx]:
                lo = self._sorted_re[idx - 1]
                hi = self._sorted_re[idx]
                break
        lo_point = self._sample_single(airfoil, lo, alpha_deg)
        hi_point = self._sample_single(airfoil, hi, alpha_deg)
        w = (reynolds - lo) / max(hi - lo, 1e-12)
        cl = (1.0 - w) * lo_point.cl + w * hi_point.cl
        cd = (1.0 - w) * lo_point.cd + w * hi_point.cd
        return PolarPoint(cl=float(cl), cd=float(max(cd, 1e-05)))

    def get_airfoil_coordinates(self, airfoil):
        if airfoil in self._coords_cache:
            return self._coords_cache[airfoil]
        cache_file = self.cache_dir / f'{_sanitize_airfoil_name(airfoil)}_coords.csv'
        if cache_file.exists():
            coords = pd.read_csv(cache_file)
        else:
            coords = self._generate_coordinates(airfoil)
            coords.to_csv(cache_file, index=False)
        coords = coords.reset_index(drop=True)
        if coords.empty:
            raise RuntimeError(f"No coordinates found for airfoil '{airfoil}'.")
        self._coords_cache[airfoil] = coords
        return coords

    def _sample_single(self, airfoil, re_bin, alpha_deg):
        polar = self._load_or_generate(airfoil, re_bin)
        alpha = polar['alpha'].to_numpy()
        cl_data = polar['cl'].to_numpy()
        cd_data = polar['cd'].to_numpy()
        cl = float(np.interp(alpha_deg, alpha, cl_data, left=cl_data[0], right=cl_data[-1]))
        cd = float(np.interp(alpha_deg, alpha, cd_data, left=cd_data[0], right=cd_data[-1]))
        return PolarPoint(cl=cl, cd=max(cd, 1e-05))

    def _cache_file(self, airfoil, re_bin):
        token = _sanitize_airfoil_name(airfoil)
        return self.cache_dir / f'{token}_re_{int(round(re_bin))}.csv'

    def _closest_re_bin(self, reynolds):
        log_re = math.log(max(reynolds, 1.0))
        return min(self._sorted_re, key=lambda v: abs(math.log(v) - log_re))

    def _resolve_backend(self):
        requested = str(self.cfg.backend).strip().lower()
        has_exec = shutil.which(self.cfg.executable) is not None
        if requested == 'surrogate':
            return 'surrogate'
        if requested == 'xfoil':
            if has_exec:
                return 'xfoil'
            if self.cfg.fallback_to_surrogate:
                self._warn('XFOIL executable was not found. Falling back to built-in surrogate backend; results are useful for demos but lower-fidelity than real XFOIL polars.')
                return 'surrogate'
            raise RuntimeError(f"XFOIL backend requested, but executable '{self.cfg.executable}' was not found on PATH.")
        if has_exec:
            return 'xfoil'
        self._warn('XFOIL executable was not found. Using built-in surrogate backend; install XFOIL for higher-fidelity aerodynamic polars.')
        return 'surrogate'

    def _warn(self, msg):
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def _load_or_generate(self, airfoil, re_bin):
        key = (airfoil, float(re_bin))
        if key in self._polar_cache:
            return self._polar_cache[key]
        cache_file = self._cache_file(airfoil, re_bin)
        if cache_file.exists():
            polar = pd.read_csv(cache_file)
        else:
            if self._backend == 'surrogate':
                polar = self._surrogate_polar(airfoil=airfoil, reynolds=re_bin)
            else:
                try:
                    polar = self._run_xfoil(airfoil=airfoil, reynolds=re_bin)
                except Exception as exc:
                    if not self.cfg.fallback_to_surrogate:
                        raise
                    self._warn(f'XFOIL failed for this run and fallback is enabled. Using surrogate polar for airfoil={airfoil}, Re={re_bin:.0f}. Original error: {exc}')
                    polar = self._surrogate_polar(airfoil=airfoil, reynolds=re_bin)
            polar.to_csv(cache_file, index=False)
        polar = polar.sort_values('alpha').drop_duplicates('alpha').reset_index(drop=True)
        if polar.empty:
            raise RuntimeError(f'Aerodynamic model returned empty polar for {airfoil} @ Re={re_bin:.0f}')
        self._polar_cache[key] = polar
        return polar

    def _generate_coordinates(self, airfoil):
        airfoil_path = Path(airfoil).expanduser()
        if airfoil_path.exists() and airfoil_path.is_file():
            parsed = self._parse_coords_file(airfoil_path)
            if not parsed.empty:
                return parsed
        if self._backend == 'surrogate':
            return self._surrogate_coordinates(airfoil)
        try:
            return self._run_xfoil_coordinates(airfoil)
        except Exception as exc:
            if not self.cfg.fallback_to_surrogate:
                raise
            self._warn(f'XFOIL coordinate export failed and fallback is enabled. Using surrogate coordinates for airfoil={airfoil}. Original error: {exc}')
            return self._surrogate_coordinates(airfoil)

    def _surrogate_polar(self, airfoil, reynolds):
        m, _p, t = _airfoil_shape_params(airfoil)
        alphas = np.arange(self.cfg.alpha_start_deg, self.cfg.alpha_end_deg + 0.5 * self.cfg.alpha_step_deg, self.cfg.alpha_step_deg, dtype=float)
        if alphas.size == 0:
            alphas = np.array([self.cfg.alpha_start_deg, self.cfg.alpha_end_deg], dtype=float)
        re_factor = (max(reynolds, 80000.0) / 300000.0) ** 0.16
        cl_slope = 2.0 * math.pi * (1.0 - 0.12 * (t - 0.12))
        cl_max = max(0.9, 1.15 + 6.5 * m + 0.8 * (0.14 - t))
        stall_alpha = max(9.0, 12.5 + 1.4 * (0.12 - t) / 0.04)
        rows = []
        for alpha in alphas:
            alpha_bias = alpha + 42.0 * m
            cl_lin = cl_slope * math.radians(alpha_bias)
            cl = cl_max * math.tanh(cl_lin / max(cl_max, 1e-06))
            stall_soft = 1.0 + (abs(alpha) / max(stall_alpha, 1e-06)) ** 4
            cl /= stall_soft ** 0.09
            cd0 = 0.0075 * (1.0 / max(re_factor, 0.35))
            cd0 += 0.014 * (t - 0.11) ** 2
            cd0 += 0.02 * (m - 0.02) ** 2
            k = 0.011 + 0.018 * max(0.0, 0.13 - t)
            cd = cd0 + k * cl * cl + 0.0014 * (alpha / 18.0) ** 4
            rows.append((float(alpha), float(cl), float(max(cd, 1e-05))))
        return pd.DataFrame(rows, columns=['alpha', 'cl', 'cd'])

    def _surrogate_coordinates(self, airfoil):
        m, p, t = _airfoil_shape_params(airfoil)
        beta = np.linspace(0.0, math.pi, 161)
        x = 0.5 * (1.0 - np.cos(beta))
        yt = 5.0 * t * (0.2969 * np.sqrt(np.maximum(x, 1e-08)) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        if m > 1e-09 and 1e-06 < p < 1.0 - 1e-06:
            mask = x < p
            yc[mask] = m / p ** 2 * (2.0 * p * x[mask] - x[mask] ** 2)
            yc[~mask] = m / (1.0 - p) ** 2 * (1.0 - 2.0 * p + 2.0 * p * x[~mask] - x[~mask] ** 2)
            dyc_dx[mask] = 2.0 * m / p ** 2 * (p - x[mask])
            dyc_dx[~mask] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~mask])
        theta = np.arctan(dyc_dx)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
        return pd.DataFrame({'x': x_coords, 'y': y_coords})

    def _run_xfoil(self, airfoil, reynolds):
        with tempfile.TemporaryDirectory(prefix='xfoil_run_') as tmpdir:
            run_dir = Path(tmpdir)
            polar_path = run_dir / 'polar.out'
            command_script = self._build_xfoil_script(airfoil=airfoil, reynolds=reynolds, polar_filename=polar_path.name)
            proc = subprocess.run([self.cfg.executable], input=command_script, text=True, capture_output=True, cwd=run_dir, timeout=self.cfg.timeout_s, check=False)
            if proc.returncode != 0 and (not polar_path.exists()):
                stderr_excerpt = proc.stderr.strip().splitlines()[-8:]
                raise RuntimeError(f'XFOIL failed for airfoil={airfoil}, Re={reynolds:.0f}. Exit={proc.returncode}. Stderr tail={stderr_excerpt}')
            if not polar_path.exists():
                stdout_excerpt = proc.stdout.strip().splitlines()[-12:]
                raise RuntimeError(f'XFOIL did not write polar file for airfoil={airfoil}, Re={reynolds:.0f}. Stdout tail={stdout_excerpt}')
            polar = self._parse_polar_file(polar_path)
            if len(polar) < 6:
                raise RuntimeError(f'Not enough valid polar points for {airfoil} @ Re={reynolds:.0f}. Points={len(polar)}')
            return polar

    def _run_xfoil_coordinates(self, airfoil):
        with tempfile.TemporaryDirectory(prefix='xfoil_coords_') as tmpdir:
            run_dir = Path(tmpdir)
            coords_path = run_dir / 'coords.dat'
            script = self._build_coords_script(airfoil=airfoil, coords_filename=coords_path.name)
            proc = subprocess.run([self.cfg.executable], input=script, text=True, capture_output=True, cwd=run_dir, timeout=self.cfg.timeout_s, check=False)
            if proc.returncode != 0 and (not coords_path.exists()):
                stderr_excerpt = proc.stderr.strip().splitlines()[-8:]
                raise RuntimeError(f'XFOIL failed while exporting coordinates for airfoil={airfoil}. Exit={proc.returncode}. Stderr tail={stderr_excerpt}')
            if not coords_path.exists():
                stdout_excerpt = proc.stdout.strip().splitlines()[-12:]
                raise RuntimeError(f'XFOIL did not write coordinate file for airfoil={airfoil}. Stdout tail={stdout_excerpt}')
            coords = self._parse_coords_file(coords_path)
            if len(coords) < 20:
                raise RuntimeError(f'Not enough coordinate points for airfoil={airfoil}. Points={len(coords)}')
            return coords

    def _build_xfoil_script(self, airfoil, reynolds, polar_filename):
        af = airfoil.strip().upper().replace('NACA', '').strip()
        if af.isdigit():
            load_line = f'NACA {af}'
        else:
            af_path = Path(airfoil).expanduser().resolve()
            load_line = f'LOAD {af_path}'
        return f'{load_line}\nPANE\nOPER\nVISC {int(round(reynolds))}\nMACH 0.0\nVPAR\nN {self.cfg.ncrit}\n\nITER {int(self.cfg.max_iter)}\nPACC\n{polar_filename}\n\nASEQ {self.cfg.alpha_start_deg:.3f} {self.cfg.alpha_end_deg:.3f} {self.cfg.alpha_step_deg:.3f}\nPACC\n\nQUIT\n'

    def _build_coords_script(self, airfoil, coords_filename):
        af = airfoil.strip().upper().replace('NACA', '').strip()
        if af.isdigit():
            load_line = f'NACA {af}'
        else:
            af_path = Path(airfoil).expanduser().resolve()
            load_line = f'LOAD {af_path}'
        return f'{load_line}\nPANE\nPSAV\n{coords_filename}\nQUIT\n'

    @staticmethod
    def _parse_polar_file(path):
        rows = []
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            cols = line.strip().split()
            if len(cols) < 3:
                continue
            if not (_looks_like_number(cols[0]) and _looks_like_number(cols[1]) and _looks_like_number(cols[2])):
                continue
            alpha = float(cols[0])
            cl = float(cols[1])
            cd = float(cols[2])
            if not math.isfinite(alpha) or not math.isfinite(cl) or (not math.isfinite(cd)):
                continue
            rows.append((alpha, cl, max(cd, 1e-05)))
        return pd.DataFrame(rows, columns=['alpha', 'cl', 'cd'])

    @staticmethod
    def _parse_coords_file(path):
        rows = []
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            cols = line.strip().split()
            if len(cols) < 2:
                continue
            if not (_looks_like_number(cols[0]) and _looks_like_number(cols[1])):
                continue
            x = float(cols[0])
            y = float(cols[1])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            rows.append((x, y))
        return pd.DataFrame(rows, columns=['x', 'y'])
