# $Id$
from setuptools import setup
from distutils.core import Extension

from kookutils import get_svnversion
svnversion = get_svnversion()
version = '0.1.dev%s'%svnversion

import os, glob, time, sys, StringIO

# flydra stuff
BUILD_FLYDRA_ARENA = False # test for comedilib below

if sys.platform.startswith('linux'):
    if (os.path.exists('/usr/local/include/comedilib.h') or
        os.path.exists('/usr/include/comedilib.h')):
        BUILD_FLYDRA_ARENA = True

install_requires = ['FlyMovieFormat','cam_iface','FastImage','wxglvideo']

FAKEIPP = False
if FAKEIPP:
    already_built_with_scons = os.path.exists('fakeipp/libfakeipp.a')
    ipp_sources = []
    ipp_include_dirs = ['fakeipp/include']
    ipp_library_dirs = []
    ipp_libraries = []
    ipp_extra_compile_args = []
    
    if already_built_with_scons:
        ipp_library_dirs = ['fakeipp']
        ipp_libraries = ['fakeipp']
    else:
        ipp_sources = glob.glob('fakeipp/src/*.c')
        if os.name == 'posix':
            # assume we're using gcc... probably a better way to test...
            # also assume we've got sse2
            ipp_extra_compile_args = ['-march=pentium4',
                                      #'-msse2',
                                      #'-mfpmath=sse',
                                      #'-funroll-loops',
                                      ]
        
else:
    ipp_include_dirs = ['/opt/intel/ipp40/include/']
    if 1:
        ipp_library_dirs = ['/opt/intel/ipp40/sharedlib/',
                            '/opt/intel/ipp40/sharedlib/linux32']
        ipp_libraries = ['ippcore',
                         'ippi',
                         'ipps',
                         'ippcv',
                         'guide']
    else:
        ipp_library_dirs = ['/opt/intel/ipp40/lib/']
        ipp_libraries = ['ippcoremerged',
                         'ippimerged',
                         'ippsmerged',
                         'ippcvmerged',
                         'guidemerged']
    ipp_sources = []
    ipp_extra_compile_args = []
    if not os.path.exists('/opt/intel/ipp40'):
        print 'WARNING: no IPP present.'

ext_modules = []


if 1:
    # Pyrex build of realtime_image_analysis
    realtime_image_analysis_extension_name='flydra.realtime_image_analysis4'
    realtime_image_analysis_sources=['src/realtime_image_analysis4.pyx',
                                     'src/c_fit_params.c',
                                     'src/eigen.c',
                                     ]+ipp_sources
    ext_modules.append(Extension(name=realtime_image_analysis_extension_name,
                                 sources=realtime_image_analysis_sources,
                                 include_dirs=ipp_include_dirs,
                                 library_dirs=ipp_library_dirs,
                                 libraries=ipp_libraries, # + ['comedi'],
                                 extra_compile_args=ipp_extra_compile_args,
                                 ))
    
ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

if BUILD_FLYDRA_ARENA:
    arena_control_extension_name='flydra.ArenaController'
    arena_control_sources=['src/ArenaController.pyx',
                           'src/arena_control.c',
                           'src/arena_feedback.c',
                           'src/arena_utils.c',
                           'src/serial_comm/serial_comm.c',
                           ]
    arena_control_libraries = ['comedi','rt']
    ext_modules.append(Extension(name=arena_control_extension_name,
                                 sources=arena_control_sources,
                                 libraries=arena_control_libraries,
                                 ))
    ext_modules.append(Extension(name='flydra.scomm',
                                 sources=['src/serial_comm/scomm.pyx',
                                          'src/serial_comm/serial_comm.c',
                                          ],
                                 ))

if os.name.startswith('posix'):
    install_requires.append('posix_sched')

setup(name='flydra',
      version=version,
      packages = ['flydra'],
      package_dir = {'flydra':'lib/flydra',
                     },
      ext_modules= ext_modules,
      
      install_requires = install_requires,
      zip_safe = True,
      )
