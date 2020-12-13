from setuptools import setup, find_packages
from distutils.core import Extension  # actually monkey-patched by setuptools
from Cython.Build import cythonize

import numpy as np

from io import open
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

ext_modules = []

ext_modules.append(
    Extension(
        name="flydra_core._reconstruct_utils",
        sources=["flydra_core/_reconstruct_utils.pyx"],
        include_dirs=[np.get_include()],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._pmat_jacobian",
        sources=["flydra_core/_pmat_jacobian.pyx"],
        include_dirs=[np.get_include()],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._pmat_jacobian_water",
        sources=["flydra_core/_pmat_jacobian_water.pyx"],
        include_dirs=[np.get_include()],
    )
)

ext_modules.append(
    Extension(
        name="flydra_core._flydra_tracked_object",
        sources=["flydra_core/_flydra_tracked_object.pyx"],
        include_dirs=[np.get_include()],
    )
)

ext_modules.append(
    Extension(name="flydra_core._mahalanobis", sources=["flydra_core/_mahalanobis.pyx"])
)

ext_modules.append(
    Extension(name="flydra_core._fastgeom", sources=["flydra_core/_fastgeom.pyx"])
)

ext_modules.append(
    Extension(
        name="flydra_core._Roots3And4",
        sources=["flydra_core/_Roots3And4.pyx", "flydra_core/Roots3And4.c",],
    )
)

ext_modules.append(
    Extension(name="flydra_core._refraction", sources=["flydra_core/_refraction.pyx"])
)

setup(
    name="flydra_core",
    version="0.7.11",  # keep in sync with flydra_core/version.py
    author="Andrew Straw",
    author_email="strawman@astraw.com",
    url="https://github.com/strawlab/flydra",
    description="flydra mainbrain and core lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    test_suite="nose.collector",
    ext_modules=cythonize(ext_modules),
    entry_points={
        "console_scripts": [
            # camera calibration
            "flydra_analysis_align_calibration = flydra_core.reconstruct:align_calibration",
            "flydra_analysis_print_cam_centers = flydra_core.reconstruct:print_cam_centers",
            "flydra_analysis_flip_calibration = flydra_core.reconstruct:flip_calibration",
        ],
    },
    package_data={"flydra_core": ["flydra_server_art.png", "sample_calibration/*",],},
)
