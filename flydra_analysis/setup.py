from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = []

ext_modules.append(
    Extension(
        name="flydra_fastfinder_help",
        sources=["flydra_fastfinder_help.pyx"],
        include_dirs=[np.get_include()],
    )
)

setup(
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
)
