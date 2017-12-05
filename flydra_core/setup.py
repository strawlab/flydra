from setuptools import setup, find_packages
from distutils.core import Extension # actually monkey-patched by setuptools
from Cython.Build import cythonize
import os
import sys

import flydra_core.version
version = flydra_core.version.__version__

import numpy as np

ext_modules = []

ext_modules.append(Extension(name='flydra_core._reconstruct_utils',
                             sources=['flydra_core/_reconstruct_utils.pyx']))

ext_modules.append(Extension(name='flydra_core._pmat_jacobian',
                             sources=['flydra_core/_pmat_jacobian.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra_core._pmat_jacobian_water',
                             sources=['flydra_core/_pmat_jacobian_water.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra_core._flydra_tracked_object',
                             sources=['flydra_core/_flydra_tracked_object.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra_core._mahalanobis',
                             sources=['flydra_core/_mahalanobis.pyx']))

ext_modules.append(Extension(name='flydra_core._fastgeom',
                             sources=['flydra_core/_fastgeom.pyx']))

ext_modules.append(Extension(name='flydra_core._Roots3And4',
                             sources=['flydra_core/_Roots3And4.pyx',
                                      'flydra_core/Roots3And4.c',
                                      ]))

ext_modules.append(Extension(name='flydra_core._refraction',
                             sources=['flydra_core/_refraction.pyx']))

setup(name='flydra_core',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='flydra mainbrain and core lib',
      packages = find_packages(),
      test_suite = 'nose.collector',
      ext_modules= cythonize(ext_modules),
      entry_points = {
    'console_scripts': [
# camera calibration
    'flydra_analysis_align_calibration = flydra_core.reconstruct:align_calibration',
    'flydra_analysis_print_cam_centers = flydra_core.reconstruct:print_cam_centers',
    'flydra_analysis_flip_calibration = flydra_core.reconstruct:flip_calibration',
    ],
      },
      package_data={'flydra_core':[
                              'flydra_server_art.png',
                              'sample_calibration/*',
                              ],
                    },
      )
