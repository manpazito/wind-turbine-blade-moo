from __future__ import annotations

from setuptools import setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build C++ extensions when possible, but do not fail installation if unavailable."""

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - platform/compiler dependent
            self.warn(f"Skipping optional C++ extension build: {exc}")

    def build_extension(self, ext) -> None:  # type: ignore[override]
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - platform/compiler dependent
            self.warn(f"Failed to build optional extension {ext.name}: {exc}")


ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "wind_turbine._bem_cpp",
            ["wind_turbine/_bem_cpp.cpp"],
            cxx_std=17,
        )
    ]
    cmdclass = {"build_ext": OptionalBuildExt}
except Exception:
    # pybind11 unavailable: package still installs with Python fallback path.
    ext_modules = []
    cmdclass = {}


setup(ext_modules=ext_modules, cmdclass=cmdclass)
