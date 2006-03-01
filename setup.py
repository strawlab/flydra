# $Id$
from setuptools import setup
from distutils.core import Extension

from kookutils import get_svnversion
svnversion = get_svnversion()
version = '0.1.dev%s'%svnversion

import os, glob, time, sys, StringIO

# flydra stuff
BUILD_FLYDRA_ARENA = False # test for comedilib below

#if sys.platform.startswith('linux'):
#    if (os.path.exists('/usr/local/include/comedilib.h') or
#        os.path.exists('/usr/include/comedilib.h')):
#        BUILD_FLYDRA_ARENA = True

install_requires = ['FlyMovieFormat','cam_iface','wxglvideo']

ext_modules = []

ext_modules.append(Extension(name='flydra.reconstruct_utils',
                             sources=['src/reconstruct_utils.pyx']))

if BUILD_FLYDRA_ARENA:
    # not in "flydra" package namespace
    arena_control_extension_name='ArenaController'
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
