"""Wind turbine optimization with XFOIL + BEM."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wind-turbine-xfoil-opt")
except PackageNotFoundError:  # pragma: no cover - local source checkout
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "bem",
    "config",
    "optimizer",
    "pipeline",
    "report",
    "xfoil",
]
