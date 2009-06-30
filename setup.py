from setuptools import setup, find_packages
from distutils.core import Extension # actually monkey-patched by setuptools
import flydra.version
import numpy as np

version = flydra.version.__version__

ext_modules = []

ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

ext_modules.append(Extension(name='flydra.pmat_jacobian',
                             sources=['src/pmat_jacobian.pyx']))

ext_modules.append(Extension(name='flydra.kalman.flydra_tracked_object',
                             sources=['src/flydra_tracked_object.c'])) # auto-generate with cython

ext_modules.append(Extension(name='flydra.mahalanobis',
                             sources=['src/mahalanobis.pyx']))

ext_modules.append(Extension(name='flydra.fastgeom',
                             sources=['src/fastgeom.pyx']))

ext_modules.append(Extension(name='flydra.a2.fastfinder_help',
                             sources=['flydra/a2/fastfinder_help.c'],
                             include_dirs=[np.get_numpy_include()],
                             )) # auto-generate with cython

setup(name='flydra',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='multi-headed fly-tracking beast',
      packages = find_packages(),
      test_suite = 'nose.collector',
      ext_modules= ext_modules,
      entry_points = {
    'console_scripts': [

# running experiments
    'flydra_camera_node = flydra.camnode:main',
# benchmarking/testing
    'flydra_bench = flydra.camnode:benchmark',
    'flydra_LED_test_latency = flydra.LEDdriver.LED_test_latency:main',
    'flydra_simulator = flydra.flydra_simulator:main',

# analysis - ufmf care and feeding
    'flydra_analysis_auto_discover_ufmfs = flydra.a2.auto_discover_ufmfs:main',
    'flydra_analysis_montage_ufmfs = flydra.a2.montage_ufmfs:main',
    'flydra_analysis_retrack_movies = flydra.a2.retrack_movies:main',

# analysis - .h5 file care and feeding
    'flydra_analysis_filter_kalman_data = flydra.analysis.flydra_analysis_filter_kalman_data:main',
    'flydra_analysis_h5_shorten = flydra.a2.h5_shorten:main',
    'flydra_analysis_check_sync = flydra.kalman.kalmanize:check_sync',
    'flydra_analysis_get_clock_sync = flydra.a2.get_clock_sync:main',
    'flydra_analysis_get_2D_image_latency = flydra.a2.get_2D_image_latency:main',

# analysis - re-kalmanize
    'flydra_kalmanize = flydra.kalman.kalmanize:main',

# analysis - not yet classified
    'flydra_analysis_convert_to_mat = flydra.analysis.flydra_analysis_convert_to_mat:main',
    'flydra_analysis_plot_clock_drift = flydra.analysis.flydra_analysis_plot_clock_drift:main',
    'flydra_analysis_plot_cameras = flydra.analysis.flydra_analysis_plot_cameras:main',
    'flydra_analysis_plot_kalman_2d = flydra.a2.plot_kalman_2d:main',
    'flydra_analysis_plot_summary = flydra.a2.plot_summary:main',
    'flydra_analysis_plot_timeseries_2d_3d = flydra.a2.plot_timeseries_2d_3d:main',
    'flydra_analysis_plot_timeseries_3d = flydra.a2.plot_timeseries:main',
    'flydra_analysis_plot_top_view = flydra.a2.plot_top_view:main',
    'flydra_analysis_print_camera_summary = flydra.analysis.flydra_analysis_print_camera_summary:main',
    'flydra_analysis_save_movies_overlay = flydra.a2.save_movies_overlay:main',
    'flydra_images_export = flydra.a2.flydra_images_export:main',
    'kdviewer = flydra.a2.kdviewer:main',
    'kdmovie_saver = flydra.a2.kdmovie_saver:main',
    'flydra_analysis_data2smoothed = flydra.a2.data2smoothed:main',
    'flydra_textlog2csv = flydra.a2.flydra_textlog2csv:main',
    'flydra_analysis_print_kalmanize_makefile_location = flydra.a2.print_kalmanize_makefile_location:main',

# analysis - image based orientation
    'flydra_analysis_image_based_orientation = flydra.a2.image_based_orientation:main',
    'flydra_analysis_orientation_ekf_fitter = flydra.a2.orientation_ekf_fitter:main',
    'flydra_analysis_orientation_ekf_plot = flydra.a2.orientation_ekf_fitter:plot_ori_command_line',
    'flydra_analysis_orientation_is_fit = flydra.a2.orientation_ekf_fitter:is_orientation_fit_sysexit',

# upload firmware to USB devices
    'flydra_LED_driver_enter_dfu_mode = flydra.LEDdriver.LEDdriver:enter_dfu_mode',
    'flydra_trigger_enter_dfu_mode = flydra.trigger:enter_dfu_mode',

# trigger device
    'flydra_trigger_check_device = flydra.trigger:check_device',
    'flydra_trigger_set_frequency = flydra.trigger:set_frequency',
    'flydra_trigger_trigger_once = flydra.trigger:trigger_once',
    'flydra_trigger_latency_measure = flydra.trigger.latency_measure:main',

# camera calibration
    'flydra_analysis_calibration_export = flydra.analysis.export_calibration:main',
    'flydra_analysis_calibration_align_gui = flydra.a2.calibration_align_gui:main',
    'flydra_analysis_generate_recalibration = flydra.analysis.flydra_analysis_generate_recalibration:main',
    'flydra_analysis_align_calibration = flydra.reconstruct:align_calibration',
    'flydra_analysis_plot_calibration_input = flydra.a2.plot_calibration_input:main',
    'flydra_analysis_calibration_to_xml = flydra.a2.calibration_to_xml:main',

# camera calibration - radial distortion stuff
    'flydra_visualize_distortions = flydra.radial_distortion.visualize_distortions:main',
    'flydra_checkerboard = flydra.radial_distortion.checkerboard:main',

# testing
    'flydra_test_commands = flydra.test_commands:main',
    'flydra_test_mpl_markersize = flydra.test.mpl_markersize:main',
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
                              'autogenerated/*',
                              ],
                    'flydra.a2':['kdmovie_saver_default_path.kmp',
                                 'sample_*.h5',
                                 'sample_*.mat',
                                 'sample_calibration.xml',
                                 'Makefile.kalmanize',
                                 ],
                    },
      )
