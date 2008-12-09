Data analysis
*************

Types of data files
===================

.. _data_analysis-file_types-calibration_files:

Calibration files
-----------------

(See :ref:`calibration` for an overview on calibration.)

Calibrations may be saved as:

 * A calibration directory (see below)

 * An .xml file with just the calibration.

Additionally, calibrations may be saved in:

 * .h5 files

 * .xml files with stimulus or trajectory information.

Calibration directories
.......................

To provide compatibility with the `Multi Camera Self Calibration
Toolbox`_ by Svoboda, et al, the calibration directory includes the
following files:

 * calibration_units.txt - a string describing the units of the calibration
 * camera_order.txt - a list of the flydra cam_ids
 * !IdMat.dat - booleans indicating the valid data
 * original_cam_centers.dat - TODO
 * points.dat - the 2D image coordinates of the calibration object
 * Res.dat - the resolution of the cameras, in pixels

The `Multi Camera Self Calibration Toolbox`_ (written in MATLAB) adds
several more files. These files contain the extrinsic and intrinsic
parameters of a pinhole model as well as non-linear terms for a radial
distortion. flydra_mainbrain loads these files and sends some of this
information to the camera nodes. Specifically the files are:

 * TODO - (Need to add here.)

.. _Multi Camera Self Calibration Toolbox: http://cmp.felk.cvut.cz/%7Esvoboda/SelfCal/index.html

Image files
-----------

 * .fmf (`Fly Movie Format`_) files contain raw, uncompressed image
   data and timestamps. Additionally, floating point formats may be
   stored to facilitate saving ongoing pixel-by-pixel mean and
   variance calculations.

 * .ufmf (micro Fly Movie Format) files contain small regions of the
   entire image. When most of the image is unchanging, this format
   allows reconstruction of near-lossless movies at a fraction of the
   disk and CPU usage of other formats.

.. _Fly Movie Format: http://code.astraw.com/projects/motmot

.. _data_analysis-tracking_data_files:

Tracking data files
-------------------

Tracking data (position over time) is stored using the `HDF5 file
format`_ using the pytables_ library. At a high level, there are two
types of such files:

 * raw 2D data files. These contain a ``/data2d_distorted``
   table. Throughout the documentation, such files will be represented
   as :file:`DATAFILE2D.h5`.

 * "Kalmanized" 3D data files. These contain ``/kalman_estimates`` and
   ``/kalman_observations`` tables in addition to a ``/calibration``
   group.  Throughout the documentation, such files will be
   represented as :file:`DATAFILE3D.h5`.

Note that a single .h5 file may have both sets of features, and thus
may be both a raw 2D data file in addition to a kalmanized 3D data
file. For any data processing step, it is usually only one or the
other aspect of this file that is important, and thus the roles above
could be played by a single file.

.. _HDF5 file format: http://www.hdfgroup.org/HDF5/index.html
.. _pytables: http://pytables.org

Stimulus, arena, and compound trajectory descriptions
-----------------------------------------------------

XML files may be used to specify many aspects of an experiment and
analysis through an extensible format. Like
:ref:`data_analysis-tracking_data_files`, these XML files may have
multiple roles within a single file type. The roles include

 * Arena description. The tags ``cubic_arena`` and
   ``cylindrical_arena``, for example, are used to define a
   rectangular and cylindrical arena, respectively.

 * Stimulus description. The tag ``cylindrical_post``, for example, is
   used to define a stationary cylinder.

Because the format is extensible, adding further support can be done
in a backwards-compatible way. These XML files are handled primarily
through :mod:`flydra.a2.xml_stimulus`.

Predefined analysis programs
============================

(TODO: port the list of programs from the webpage.)

Automating data analysis
========================

The module :mod:`flydra.a2.flydra_scons` provides definitions that may
be useful in building SConstruct files for scons_. Using scons allows
relatively simple batch processing to be specified, including the
ability to concurrently execute several jobs at once.

.. _scons: http://scons.org

Source code for your own data analysis
======================================

The module :mod:`flydra.a2.core_analysis` has fast, optimized
trajectory opening routines.
