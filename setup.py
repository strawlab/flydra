# $Id$
from setuptools import setup
from distutils.core import Extension

from motmot.utils.utils import get_svnversion_persistent
version_str = '0.3.dev%(svnversion)s'
version = get_svnversion_persistent('flydra/version.py',version_str)

ext_modules = []

ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

ext_modules.append(Extension(name='flydra.pmat_jacobian',
                             sources=['src/pmat_jacobian.pyx']))

ext_modules.append(Extension(name='flydra.mahalanobis',
                             sources=['src/mahalanobis.pyx']))

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
                  'flydra.trigger',
                  'flydra.LEDdriver',
                  'flydra.radial_distortion',
                  ],
      ext_modules= ext_modules,
      entry_points = {
    'console_scripts': [
    'flydra_camera_node = flydra.camnode:main',
    'flydra_bench = flydra.flydra_bench:main',
    'flydra_kalmanize = flydra.kalman.kalmanize:main',

    'flydra_analysis_convert_to_mat = flydra.analysis.flydra_analysis_convert_to_mat:main',
    'flydra_analysis_filter_kalman_data = flydra.analysis.flydra_analysis_filter_kalman_data:main',
    'flydra_analysis_generate_recalibration = flydra.analysis.flydra_analysis_generate_recalibration:main',
    'flydra_analysis_plot_clock_drift = flydra.analysis.flydra_analysis_plot_clock_drift:main',
    'flydra_analysis_plot_cameras = flydra.analysis.flydra_analysis_plot_cameras:main',
    'flydra_analysis_plot_kalman_2d = flydra.a2.plot_kalman_2d:main',
    'flydra_analysis_plot_calibration_input = flydra.a2.plot_calibration_input:main',
    'flydra_analysis_calibration_to_xml = flydra.a2.calibration_to_xml:main',
    'flydra_analysis_plot_timeseries_2d_3d = flydra.a2.plot_timeseries_2d_3d:main',
    'flydra_analysis_plot_timeseries_3d = flydra.a2.plot_timeseries:main',
    'flydra_analysis_plot_top_view = flydra.a2.plot_top_view:main',
    'flydra_analysis_print_camera_summary = flydra.analysis.flydra_analysis_print_camera_summary:main',
    'flydra_analysis_save_movies_overlay = flydra.a2.save_movies_overlay:main',

    'flydra_LED_driver_enter_dfu_mode = flydra.LEDdriver.LEDdriver:enter_dfu_mode',
    'flydra_LED_test_latency = flydra.LEDdriver.LED_test_latency:main',

    'flydra_trigger_enter_dfu_mode = flydra.trigger:enter_dfu_mode',
    'flydra_trigger_check_device = flydra.trigger:check_device',
    'flydra_trigger_set_frequency = flydra.trigger:set_frequency',
    'flydra_trigger_trigger_once = flydra.trigger:trigger_once',
    'flydra_trigger_latency_test = flydra.trigger.latency_test:main',

    'flydra_images_export = flydra.a2.flydra_images_export:main',
    'flydra_visualize_distortions = flydra.radial_distortion.visualize_distortions:main',

    'kdviewer = flydra.a2.kdviewer:main',
    'kdmovie_saver = flydra.a2.kdmovie_saver:main',
    'data2smoothed = flydra.a2.data2smoothed:main',
    'flydra_textlog2csv = flydra.a2.flydra_textlog2csv:main',

    'flydra_checkerboard = flydra.radial_distortion.checkerboard:main',
    ],
    'gui_scripts': [
    'flydra_mainbrain = flydra.wxMainBrain:main',
    ],
    'flydra.kdviewer.plugins':['default = flydra.a2.conditions_draw:default',
                               'mama07 = flydra.a2.conditions_draw:mama07',
                               'mama20080414 = flydra.a2.conditions_draw:mama20080414',
                               'mama20080501 = flydra.a2.conditions_draw:mama20080501',
                               'hum07 = flydra.a2.conditions_draw:hum07',
                               'wt0803 = flydra.a2.conditions_draw:wt0803',
                               ],
    },
      zip_safe = False, # must be false for flydra_bench
      package_data={'flydra':['flydra_server.xrc',
                              'flydra_server_art.png',
                              'detect.wav',
                              'sample_calibration/*',
                              ],
                    'flydra.a2':['kdmovie_saver_default_path.kmp',
                                 'sample_*.h5',
                                 'sample_*.mat',
                                 ],
                    },
      )
