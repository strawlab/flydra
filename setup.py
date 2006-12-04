# $Id$
from setuptools import setup
from distutils.core import Extension

from kookutils import get_svnversion_persistent
version_str = '0.2.dev%(svnversion)s'
version = get_svnversion_persistent('flydra/version.py',version_str)

import os, glob, time, sys, StringIO

ext_modules = []

ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

setup(name='flydra',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='multi-headed fly-tracking beast',
      packages = ['flydra',
                  'flydra.kalman',
                  'flydra.analysis',
                  'flydra.trigger'],
      ext_modules= ext_modules,
      entry_points = {
    'console_scripts': [
    'flydra_camera_node = flydra.flydra_camera_node:main',
    'flydra_bench = flydra.flydra_bench:main',
    'flydra_kalmanize = flydra.kalman.kalmanize:main',
    'flydra_analysis_convert_to_mat = flydra.analysis.flydra_analysis_convert_to_mat:main',
    'flydra_analysis_plot_clock_drift = flydra.analysis.flydra_analysis_plot_clock_drift:main',
    'flydra_analysis_plot_cameras = flydra.analysis.flydra_analysis_plot_cameras:main',
    'flydra_analysis_plot_kalman_data = flydra.analysis.flydra_analysis_plot_kalman_data:main',
    'flydra_analysis_plot_kalman_2d = flydra.analysis.flydra_analysis_plot_kalman_2d:main',
    'flydra_trigger_enter_dfu_mode = flydra.trigger:enter_dfu_mode',
    'flydra_trigger_check_device = flydra.trigger:check_device',
    ],
    'gui_scripts': [
    'flydra_mainbrain = flydra.wxMainBrain:main',
    ],
    },
      zip_safe = False, # must be false for flydra_bench
      package_data={'flydra':['flydra_server.xrc',
                              'flydra_server_art.png',
                              'detect.wav',
                              ],
                    },
      )
