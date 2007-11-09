# $Id$
from setuptools import setup
from distutils.core import Extension

from motmot_utils import get_svnversion_persistent
version_str = '0.3.dev%(svnversion)s'
version = get_svnversion_persistent('flydra/version.py',version_str)

ext_modules = []

ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

ext_modules.append(Extension(name='flydra.fastgeom',
                             sources=['src/fastgeom.pyx']))

setup(name='flydra',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='multi-headed fly-tracking beast',
      packages = ['flydra',
                  'flydra.kalman',
                  'flydra.analysis',
                  'flydra.a2', # new analysis
                  'flydra.trigger'],
      ext_modules= ext_modules,
      entry_points = {
    'console_scripts': [
    'flydra_camera_node = flydra.flydra_camera_node:main',
    'flydra_bench = flydra.flydra_bench:main',
    'flydra_kalmanize = flydra.kalman.kalmanize:main',
    
    'flydra_analysis_convert_to_mat = flydra.analysis.flydra_analysis_convert_to_mat:main',
    'flydra_analysis_generate_recalibration = flydra.analysis.flydra_analysis_generate_recalibration:main',
    'flydra_analysis_plot_clock_drift = flydra.analysis.flydra_analysis_plot_clock_drift:main',
    'flydra_analysis_plot_cameras = flydra.analysis.flydra_analysis_plot_cameras:main',
    'flydra_analysis_plot_kalman_data = flydra.analysis.flydra_analysis_plot_kalman_data:main',
    'flydra_analysis_plot_kalman_2d = flydra.analysis.flydra_analysis_plot_kalman_2d:main',
    'flydra_analysis_print_camera_summary = flydra.analysis.flydra_analysis_print_camera_summary:main',
    
    'flydra_trigger_enter_dfu_mode = flydra.trigger:enter_dfu_mode',
    'flydra_trigger_check_device = flydra.trigger:check_device',
    'flydra_trigger_set_frequency = flydra.trigger:set_frequency',
    'flydra_trigger_trigger_once = flydra.trigger:trigger_once',
    'flydra_trigger_latency_test = flydra.trigger.latency_test:main',

    'kdviewer = flydra.a2.kdviewer:main',
    'kdmovie_saver = flydra.a2.kdmovie_saver:main',
    'data2smoothed = flydra.a2.data2smoothed:main',
    ],
    'gui_scripts': [
    'flydra_mainbrain = flydra.wxMainBrain:main',
    ],
    },
      zip_safe = False, # must be false for flydra_bench
      package_data={'flydra':['flydra_server.xrc',
                              'flydra_server_art.png',
                              'detect.wav',
                              'sample_calibration/*',
                              ],
                    'flydra.a2':['kdmovie_saver_default_path.kmp'],
                    },
      )
