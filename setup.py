# A special note about building a source package: just do it with git archive.

from setuptools import setup, find_packages
from distutils.core import Extension # actually monkey-patched by setuptools
from Cython.Build import cythonize
import os
import sys

import flydra.version
version = flydra.version.__version__

import numpy as np

ext_modules = []

LIGHT_INSTALL = int(os.environ.get('LIGHT_INSTALL','0'))

if not LIGHT_INSTALL:
    import motmot.FastImage.FastImage # set LIGHT_INSTALL env variable to skip
    FastImage = motmot.FastImage.FastImage

    import motmot.FastImage.util as FastImage_util
    IPPROOT = os.environ['IPPROOT']

    # build with same IPP as FastImage
    vals = FastImage_util.get_build_info(ipp_static=False, # use dynamic linking
                                         ipp_arch=FastImage.get_IPP_arch(),
                                         ipp_root=IPPROOT,
                                         )

    ext_modules.append(Extension(name='flydra.camnode_colors',
                                 sources=['flydra/camnode_colors.pyx','flydra/colors.c'],
                                 include_dirs=vals['ipp_include_dirs'],
                                 library_dirs=vals['ipp_library_dirs'],
                                 libraries=vals['ipp_libraries'],
                                 define_macros=vals['ipp_define_macros'],
                                 extra_link_args=vals['extra_link_args'],
                                 ))

ext_modules.append(Extension(name='flydra._reconstruct_utils',
                             sources=['flydra/_reconstruct_utils.pyx']))

ext_modules.append(Extension(name='flydra._pmat_jacobian',
                             sources=['flydra/_pmat_jacobian.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra._pmat_jacobian_water',
                             sources=['flydra/_pmat_jacobian_water.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra._flydra_tracked_object',
                             sources=['flydra/_flydra_tracked_object.pyx'],
                             include_dirs=[np.get_include()],
                         ))

ext_modules.append(Extension(name='flydra._mahalanobis',
                             sources=['flydra/_mahalanobis.pyx']))

ext_modules.append(Extension(name='flydra._fastgeom',
                             sources=['flydra/_fastgeom.pyx']))

ext_modules.append(Extension(name='flydra._Roots3And4',
                             sources=['flydra/_Roots3And4.pyx',
                                      'flydra/Roots3And4.c',
                                      ]))

ext_modules.append(Extension(name='flydra._refraction',
                             sources=['flydra/_refraction.pyx']))

ext_modules.append(Extension(name='flydra.a2.fastfinder_help',
                             sources=['flydra/a2/fastfinder_help.pyx'],
                             include_dirs=[np.get_include()],
                             ))

setup(name='flydra',
      version=version,
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      description='multi-headed fly-tracking beast',
      packages = find_packages(),
      test_suite = 'nose.collector',
      ext_modules= cythonize(ext_modules),
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

# analysis - generate movies with tracking overlays (uses fmfs or ufmfs)
    'flydra_analysis_overlay_kalman_movie = flydra.a2.overlay_kalman_movie:main',

# analysis - .h5 file care and feeding
    'flydra_analysis_print_h5_info = flydra.a2.h5_info:cmdline',
    'flydra_analysis_filter_kalman_data = flydra.analysis.flydra_analysis_filter_kalman_data:main',
    'flydra_analysis_h5_shorten = flydra.a2.h5_shorten:main',
    'flydra_analysis_check_mainbrain_h5_contiguity = flydra.a2.check_mainbrain_h5_contiguity:main',
    'flydra_analysis_check_sync = flydra.kalman.kalmanize:check_sync',
    'flydra_analysis_get_clock_sync = flydra.a2.get_clock_sync:main',
    'flydra_analysis_get_2D_image_latency = flydra.a2.get_2D_image_latency:main',
    'flydra_analysis_get_2D_image_latency_plot = flydra.a2.get_2D_image_latency_plot:main',

# analysis - re-kalmanize
    'flydra_kalmanize = flydra.kalman.kalmanize:main',

# timestamp conversion
    'flydra_analysis_frame2timestamp = flydra.analysis.result_utils:frame2timestamp_command',
    'flydra_analysis_timestamp2frame = flydra.analysis.result_utils:timestamp2frame_command',


# analysis - not yet classified
    'flydra_analysis_convert_to_mat = flydra.analysis.flydra_analysis_convert_to_mat:main',
    'flydra_analysis_plot_clock_drift = flydra.analysis.flydra_analysis_plot_clock_drift:main',
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
    'flydra_analysis_export_flydra_hdf5 = flydra.a2.data2smoothed:export_flydra_hdf5',
    'flydra_textlog2csv = flydra.a2.flydra_textlog2csv:main',
    'flydra_analysis_print_kalmanize_makefile_location = flydra.a2.print_kalmanize_makefile_location:main',
    'flydra_analysis_calculate_reprojection_errors = flydra.a2.calculate_reprojection_errors:main',
    'flydra_analysis_retrack_reuse_data_association = flydra.a2.retrack_reuse_data_association:main',
    'flydra_analysis_calculate_skipped_frames = flydra.a2.calculate_skipped_frames:main',
    'flydra_analysis_plot_skipped_frames = flydra.a2.plot_skipped_frames:main',

# analysis - image based orientation
    'flydra_analysis_image_based_orientation = flydra.a2.image_based_orientation:main',
    'flydra_analysis_orientation_ekf_fitter = flydra.a2.orientation_ekf_fitter:main',
    'flydra_analysis_orientation_ekf_plot = flydra.a2.orientation_ekf_plot:main',
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
    'flydra_analysis_calibration_align_gui = flydra.a2.calibration_align_gui:main',
    'flydra_analysis_generate_recalibration = flydra.analysis.flydra_analysis_generate_recalibration:main',
    'flydra_analysis_align_calibration = flydra.reconstruct:align_calibration',
    'flydra_analysis_plot_calibration_input = flydra.a2.plot_calibration_input:main',
    'flydra_analysis_calibration_to_xml = flydra.a2.calibration_to_xml:main',
    'flydra_analysis_water_surface_align = flydra.a2.water_surface_align:main',
    'flydra_analysis_print_cam_centers = flydra.reconstruct:print_cam_centers',
    'flydra_analysis_plot_camera_positions = flydra.a2.plot_camera_positions:main',
    'flydra_analysis_flip_calibration = flydra.reconstruct:flip_calibration',

# camera calibration - radial distortion stuff
    'flydra_visualize_distortions = flydra.radial_distortion.visualize_distortions:main',
    'flydra_checkerboard = flydra.radial_distortion.checkerboard:main',

# ROS pointcloud stuff
    'flydra_analysis_rosbag2flydrah5 = flydra.a2.rosbag2flydrah5:main',

# testing
    'flydra_test_commands = flydra.test_commands:main',
    'flydra_test_mpl_markersize = flydra.mpl_markersize:main',
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
