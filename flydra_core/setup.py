from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = []

ext_modules.append(
    Extension(
        name="flydra_core._reconstruct_utils",
        sources=["flydra_core/_reconstruct_utils.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._pmat_jacobian",
        sources=["flydra_core/_pmat_jacobian.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._pmat_jacobian_water",
        sources=["flydra_core/_pmat_jacobian_water.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._flydra_tracked_object",
        sources=["flydra_core/_flydra_tracked_object.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
)

ext_modules.append(
    Extension(name="flydra_core._mahalanobis", sources=["flydra_core/_mahalanobis.pyx"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],)
)

ext_modules.append(
    Extension(name="flydra_core._fastgeom", sources=["flydra_core/_fastgeom.pyx"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],)
)

ext_modules.append(
    Extension(
        name="flydra_core._Roots3And4",
        sources=["flydra_core/_Roots3And4.pyx", "flydra_core/Roots3And4.c",],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
)

ext_modules.append(
    Extension(name="flydra_core._refraction", sources=["flydra_core/_refraction.pyx"], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],)
)

setup(
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    package_data={"flydra_core": ["flydra_server_art.png", "sample_calibration/*",],},
)
