from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy



setup(
    ext_modules=cythonize(["calc_A_matrix.pyx,Aij_matrix.pyx"]),
    include_dirs=[numpy.get_include()]
)   