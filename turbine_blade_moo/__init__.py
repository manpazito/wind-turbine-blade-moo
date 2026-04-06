from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version('turbine-blade-moo')
except PackageNotFoundError:
    __version__ = '0.0.0'
__all__ = ['__version__', 'bem', 'config', 'optimizer', 'pipeline', 'report', 'xfoil']
