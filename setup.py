from setuptools import setup
from setuptools.command.build_ext import build_ext

class OptionalBuildExt(build_ext):

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self.warn(f'Skipping optional C++ extension build: {exc}')

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self.warn(f'Failed to build optional extension {ext.name}: {exc}')
ext_modules = []
cmdclass = {}
try:
    from pybind11.setup_helpers import Pybind11Extension
    ext_modules = [Pybind11Extension('turbine_blade_moo._bem_cpp', ['turbine_blade_moo/_bem_cpp.cpp'], cxx_std=17)]
    cmdclass = {'build_ext': OptionalBuildExt}
except Exception:
    ext_modules = []
    cmdclass = {}
setup(ext_modules=ext_modules, cmdclass=cmdclass)
