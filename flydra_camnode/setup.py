from setuptools import setup, find_packages
from distutils.core import Extension # actually monkey-patched by setuptools
from Cython.Build import cythonize
import os
import sys

import flydra_camnode.version
version = flydra_camnode.version.__version__

import numpy as np

ext_modules = []

if 1:
    import motmot.FastImage.FastImage
    import motmot.FastImage as fi_mod
    FastImage = motmot.FastImage.FastImage

    import motmot.FastImage.util as FastImage_util
    IPPROOT = os.environ['IPPROOT']

    vals = FastImage_util.get_build_info(ipp_arch=FastImage.get_IPP_arch(),
                                         ipp_root=IPPROOT,
                                         )

    ext_modules.append(Extension(name='flydra_camnode.camnode_colors',
                                 sources=['flydra_camnode/camnode_colors.pyx','flydra_camnode/colors.c'],
                                 include_dirs=vals['ipp_include_dirs']+[np.get_include(), fi_mod.get_include()],
                                 library_dirs=vals['ipp_library_dirs'],
                                 libraries=vals['ipp_libraries'],
                                 define_macros=vals['ipp_define_macros'],
                                 extra_link_args=vals['extra_link_args'],
                                 extra_objects=vals['ipp_extra_objects'],
                                 ))

setup(name='flydra_camnode',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='flydra camera node',
      packages = find_packages(),
      test_suite = 'nose.collector',
      ext_modules= cythonize(ext_modules),
      setup_requires=["setuptools_git >= 0.3",],
      entry_points = {
    'console_scripts': [

# running experiments
    'flydra_camera_node = flydra_camnode.camnode:main',
# benchmarking/testing
    'flydra_bench = flydra_camnode.camnode:benchmark',
    ],
      }
    )
