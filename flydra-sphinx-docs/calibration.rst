.. _calibration:

Calibration
===========

Flydra calibration data (also, in the flydra source code, a
*reconstructor*) consists of:

 * parameters for a *linear pinhole camera model* (including intrinsic
   and extrinsic calibration).

 * parameters for *non-linear distortion*.

 * (optional) a *scale factor* and *units* (e.g. 1000.0 and
   millimeters). If not specified these default to 1.0 and meters. If
   these are specified they specify how to convert the native units
   into meters (the scale factor) and the name of the present
   units. (Meters are the units used the dyanmic models, and otherwise
   have no significance.)

See :ref:`data_analysis-file_types-calibration_files` for a discussion
of the calibration file formats.

Generating a calibration
------------------------

.. This was the old method numbered "2b".

The basic idea is that you will save raw 2D data points which are easy
to track and associate with a 3D object. Typically this is done by
waving an LED around.

For this to work, the 2D/3D data association problem must be
trivial. Only a single 3D point should be generating 2D points, and
every 2D point should come from the (only) 3D point. (Missing 2D
points are OK; occlusions *per se* do not cause problems, whereas
insufficient or false data associations do.)

(3D trajectories, which can only come from having a calibration and
solving the data association problem, will not be used for this step,
even if available. If a calibration has already been used to generate
Kalman state estimates, the results of this data association will be
ignored.)


Saving the calibration data in flydra_mainbrain
...............................................

Start saving data normally within the :command:`flydra_mainbrain`
application. Remember that only time points with more than 2 cameras
returning data will be useful, and that time points with more than 1
detected view per camera are useless.

Walk to the arena and wave the LED around. Make sure it covers the
entire tracking volume, and if possible, outside the tracking volume,
too.

Exporting the data for MultiCamSelfCal
......................................

Now, you have saved an .h5 file. To export the data from it for
calibration, run::

  flydra_analysis_generate_recalibration --2d-data DATAFILE2D.h5 \
    --disable-kalman-objs DATAFILE2D.h5

You should now have a new directory named
``DATAFILE2D.h5.recal``. This contains the calibration in a format
that the MATLAB MultiCamSelfCal can understand, the calibration
directory.

.. _3d-recal:

Running MultiCamSelfCal
.......................

Edit the file ``kookaburra/MultiCamSelfCal/CommonCfgAndIO/configdata.m``.

In the ``SETUP_NAME`` section, there a few variables you probably want
to examine:

 * In particular, set ``config.paths.data`` to the directory where
   your calibration data is. This is the output of the
   ``flydra_analysis_generate_recalibration`` command. Note: this must
   end in a slash (``/``).

 * ``config.cal.GLOBAL_ITER_THR`` is the criterion threshold
   reprojection error that all cameras must meet before terminating
   the global iteration loop. Something like 0.8 will be decent for an
   initial calibration (tracking an LED), but tracking tiny
   *Drosophila* should enable you to go to 0.3 or so (in other words,
   generating calibration data with the :ref:`3d-recal` method). To
   estimate the non-linear distortion (often not necessary), set this
   small enough that ``gocal`` runs non-linear parameter estimation at
   least once. This non-linear estimation step fits the radial
   distortion term.

 * ``config.cal.USE_NTH_FRAME`` if your calibration data set is too
   massive, reduce it with this variable. Typically, a successful
   calibration will have about 300-500 points being used in the final
   calibration. The number of points used will be displayed during the
   calibration step (For example, "437 points/frames have survived
   validations so far".)

 * ``config.files.idxcams`` should be set to ``[1:X]`` where ``X`` is
   the number of cameras you are using.

The other files to consider are
``MultiCamSelfCal/CommonCfgAndIO/expname.m`` and
``kookaburra/MultiCamSelfCal/MultiCamSelfCal/BlueCLocal/SETUP_NAME.m``. The
first file returns a string that specifies the setup name, and
consequently the filename (written above as ``SETUP_NAME``) for the
second file.  This second file contains (approximate) camera positions
which are used to determine the rotation, translation, and scale
factors for the final camera calibration. The current dynamic models
operate in meters, while the flydra code automatically multiplies
post-calibration 3D coordinates by 1000 (thus, converting millimeters
to meters) unless a file named ``calibration_units.txt`` specifies the
units. Thus, unless you create this file, use millimeters for your
calibration units.

Run MATLAB (e.g. ``matlab -nodesktop -nojvm``). From the MATLAB
prompt::

  cd kookaburra/MultiCamSelfCal/MultiCamSelfCal/
  gocal

When the initial mean reprojection errors are displayed, numbers of 10
pixels or less bode pretty well for this calibration to converge. It
is rare, to get a good calibration when the first iteration has large
reprojection errors. Running on a fast computer (e.g. Core 2 Duo 2
GHz), a calibration shouldn't take more than 5 minutes before looking
pretty good if things are going well. Note that, for the first
calibration, it may not be particularly important to get a great
calibration because it will be redone due to the considerations listed
in :ref:`3d-recal`.

Advanced: using 3D trajectories to re-calibrate using MultiCamSelfCal
.....................................................................

.. This is the old method 2a.

Often, it is possible (and desirable) to make a higher precision
trajectory than that possible by waving an LED. For example, flying
*Drosophila* are smaller and therefore more precisely localized points
than an LED. Also, in setups in which cameras film through movable
transparent material, flies fly in the final experimental
configuration, which may have slightly different optics that should be
part of your final calibration.

By default, you enter previously-tracked trajectory ID numbers and the
2D data that comprised these trajectories are output.

This method also saves a directory with the raw data expected by the
Multi Camera Self Calibration Toolbox.

::

  # NOTE: if your 2D and 3D data are in one file, 
  # don't use the "--2d-data" argument.
  flydra_analysis_generate_recalibration DATAFILE3D.h5 EFILE \
     --2d-data DATAFILE2D.h5
  # This will output a new calibration directory in 
  # DATAFILE3D.h5.recal

The ``EFILE`` above should have the following format (for example)::

  # These are the obj_ids of traces to use.
  long_ids = [655, 646, 530, 714, 619, 288, 576, 645]
  # These are the obj_ids of traces not to use (exluded 
  # from the list in long_ids)
  bad=[]

Finally, run the Multi Cam Self Calibration procedure on the new
calibration directory. Lower your threshold to, e.g.,
``config.cal.GLOBAL_ITER_THR = .4;``. You might want to adjust
``config.cal.USE_NTH_FRAME`` again to get the right number of data
points. This is a precise calibration, it might take as many as 30
iterations and 15 minutes.

Aligning a calibration
----------------------

Often, even if a calibration from MultiCamSelfCal creates
reprojections with minimal error and the relative camera positions
look OK, reconstructed world coordinates do not correspond with
desired world coordinates. To align the calibration the
:command:`flydra_analysis_calibration_align_gui` program may be used::

  flydra_analysis_calibration_align_gui DATAFILE3D.h5 --stim-xml=STIMULUS.xml

This results in a GUI that looks a bit like

.. image:: screenshots/flydra_analysis_calibration_align_gui.png

Using the controls on the right, align your data such that it
corresponds with the 3D model loaded by STIMULUS.xml. When you are
satisfied, click either of the save buttons to save your newly-aligned
calibration.
