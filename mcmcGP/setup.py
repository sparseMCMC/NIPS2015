from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize([
        Extension("fastMultiClassLikelihood",
        ["fastMultiClassLikelihood.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']),
        ])
)
