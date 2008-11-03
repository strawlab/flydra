flydra documentation
====================

Flydra is a realtime, multi-camera flying animal tracking system. See
the manuscript for algorithmic details.

Contents:

.. toctree::

  modules.rst

Types of data files
===================

Calibration files
-----------------

Flydra calibration data consists of: parameters for a *linear pinhole
camera model* (including intrinsic and extrinsic calibration),
parameters for *non-linear distortion*, and *units*
(e.g. millimeters). Calibrations may be saved as:

 * A calibration directory

 * An .xml file with just the calibration.

Additionally, calibrations may be saved in:

 * .h5 files

 * .xml files with stimulus or trajectory information.

Image files
-----------

 * .fmf (`Fly Movie Format`_) files contain raw, uncompressed image
   data and timestamps. Additionally, floating point formats may be
   stored to facilitate saving ongoing pixel-by-pixel mean and
   variance calculations.

 * .ufmf (micro Fly Movie Format) files contain small regions of the
   entire image, and allow reconstruction of near-lossless movies at a
   fraction of the disk and CPU usage of other formats.

.. _Fly Movie Format: http://code.astraw.com/projects/motmot

Tracking data files
-------------------

 * raw 2D data files.

 * "Kalmanized" 3D data files.

Predefined analysis programs
============================

(TODO: port the list of programs from the webpage.)

Automating data analysis
========================

:mod:`flydra.a2.flydra_scons` provides definitions that may be useful
in building SConstruct files for scons_. Using scons allows relatively
simple batch processing to be specified, including the ability to
concurrently execute several jobs at once.

.. _scons: http://scons.org

Source code for your own data analysis
======================================

:mod:`flydra.a2.core_analysis` is fast, optimized trajectory opening routines.

See modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

