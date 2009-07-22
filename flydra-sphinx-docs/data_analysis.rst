Data analysis
*************

Types of data files
===================

.. _data_analysis-file_types-calibration_files:

Calibration files and directories - overview
--------------------------------------------

(See :ref:`calibration` for an overview on calibration.)

Calibrations may be saved as:

 * A calibration directory (see below)

 * An .xml file with just the calibration.

Additionally, calibrations may be saved in:

 * .h5 files

 * .xml files that also include stimulus or trajectory information.

All calibration sources can save the camera matrix for the linear
pinhole camera model for each camera, the scale factor and units of
the overall calibration, and the non-linear distortion parameters for
each camera.

Calibration directories
.......................

To provide compatibility with the `Multi Camera Self Calibration
Toolbox`_ by Svoboda, et al, the calibration directory includes the
following files:

 * calibration_units.txt - a string describing the units of the calibration
 * camera_order.txt - a list of the flydra cam_ids
 * IdMat.dat - booleans indicating the valid data
 * original_cam_centers.dat - TODO
 * points.dat - the 2D image coordinates of the calibration object
 * Res.dat - the resolution of the cameras, in pixels
 * camN.rad - the non-linear distortion terms for camera N

The `Multi Camera Self Calibration Toolbox`_ (written in MATLAB) adds
several more files. These files contain the extrinsic and intrinsic
parameters of a pinhole model as well as non-linear terms for a radial
distortion. flydra_mainbrain loads these files and sends some of this
information to the camera nodes. Specifically the files are:

 * TODO - (Need to write about the files that the MultiCamSelfCal
   toolbox adds here.)

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

 * raw 2D data files. These contain ``/data2d_distorted`` and
   ``/cam_info`` tables. Throughout the documentation, such files will
   be represented as :file:`DATAFILE2D.h5`.

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

Data flow
=========

.. graphviz::

  digraph G {
    size ="6,4";
    TwoDee -> da;
    cal -> da;
    motion_model -> da;
    da -> kalman_observations;
    da -> kalman_estimates;
    kalman_observations -> smoothed_kalman_estimates;
    motion_model -> smoothed_kalman_estimates;

    da [label="data association & tracking (flydra_kalmanize or flydra_mainbrain)"];
    TwoDee [label="2D observations"];
    cal [label="calibration"];
    motion_model [label="dynamic model"];
    kalman_estimates [label="kalman_estimates (in .h5 file)"];
    kalman_observations [label="kalman_observations (in .h5 file)"];
    smoothed_kalman_estimates [label="smoothed kalman estimates [output of load_data(use_kalman_smoothing=True)]"];
  }


Extracting longitudinal body orientation
========================================

Theoretical overview
--------------------

Our high-throughput automated pitch angle estimation algorithm
consists of two main steps: first, the body angle is estimated in (2D)
image coordinates for each camera view, and second, the data from
multiple cameras are fused to establish a 3D estimate of longitudinal
body orientation. We take as input the body position, the raw camera
images, and an estimate of background appearance (without the
fly). These are calculated in a previous step according to the EKF
based algorithm described in the flydra manuscript.

For the first step (2D body angle estimation), we do a background
subtraction and thresholding operation to extract a binary image
containing the silhouette of the fly. A potential difficulty is
distinguishing the portion of the silhouette caused by the wings from
the portion caused by the head, thorax, and abdomen. We found
empirically that performing a connected components analysis on the
binary image thresholded using an appropriately chosen threshold value
discriminates the wings from the body with high success. Once the body
pixels are estimated in this way, a covariance matrix of these pixels
is formed and its eigenvalues and eigenvectors are used to determine
the 2D orientation of luminance within this binary image of the fly
body. **To add:** a description of the image blending technique used
with high-framerate images for ignoring flapping wings.

From the N estimates of body angle from N camera views, an estimate of
the 3D body axis direction is made. For each camera, a line from the
2D body angle estimate on the image plane and the 3D camera center
(from the known camera calibration) are used to find a plane in 3D
space.  Then, the best-fit line of intersection of the N planes is
then found using an algorithm based on singular value decomposition.


Practical steps
---------------

Estimating longitudinal body orientation happens in several steps:

* Acquire data with good 2D tracking, a good calibration, and .ufmf
  movies in good lighting.

* Perform tracking and data assocation on the 2D data to get 3D data
  using :command:`flydra_kalmanize`.

* Run :command:`flydra_analysis_image_based_orientation` to estimate
  2D longitudinal body axis.

* Check the 2D body axis estimates using :command:`flydra_analysis_montage_ufmfs` 
  to generate images or movies of the tracking.

* Finally, another rounte through the tracker and data association now
  using the 2D orientation data.

An example of a call to
:command:`flydra_analysis_image_based_orientation` is: (This was
automatically called via an SConstruct script using
:mod:`flydra.a2.flydra_scons`.)

::

  flydra_analysis_image_based_orientation --h5=DATA20080915_164551.h5 --kalman=DATA20080915_164551.kalmanized.h5 \
    --ufmfs=small_20080915_164551_cam1_0.ufmf:small_20080915_164551_cam2_0.ufmf:small_20080915_164551_cam3_0.ufmf:small_20080915_164551_cam4_0.ufmf \
    --output-h5=DATA20080915_164551.image-based-re2d.h5

When calling :command:`flydra_analysis_montage_ufmfs`, you'll need to
use at least the following elements in a configuration file::

  [what to show]
  show_2d_orientation = True

An example output from from doing something like this is shown here:

.. image:: screenshots/image_based_angles.jpg
  :width: 538
  :height: 418

The **critical issue** is that the body orientations are well tracked
in 2D. There's nothing that can be done in later processing stages if
the 2D body angle extraction is not good.