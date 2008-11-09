.. _calibration:

Calibration
===========

Flydra calibration data consists of:

 * parameters for a *linear pinhole camera model* (including intrinsic
and extrinsic calibration).

 * parameters for *non-linear distortion*.

 * (optional) a *scale factor* and *units* (e.g. 1000.0 and
   millimeters). If not specified these default to 1.0 and meters. If
   these are specified they specify how to convert the native units
   into meters (the scale factor) and the name of the present units.

See :ref:`data_analysis-file_types-calibration_files` for a discussion
of the calibration file formats.

Generating a calibration
------------------------

Saving the calibration data in flydra_mainbrain
...............................................

Save data normally within the :command:`flydra_mainbrain`
application. Only time points with more than 2 cameras returning data
will be useful, and that time points with more than 1 detected view
per camera are useless.

Exporting the data for MultiCamSelfCal
......................................

Now, you have saved an .h5 file. To export the data from it for
calibration, run::

  flydra_analysis_calibration_export DATAFILE2D.h5

You should now have a new directory named ``DATAFILE2D.h5.recal``.

Running MultiCamSelfCal
.......................

To be written.

Aligning a calibration
----------------------

Often, even if a calibration from MultiCamSelfCal reprojects with
minimal error, it is not resulting in world coordinates corresponding
with desired world coordinates. To align the calibration the
:command:`flydra_analysis_calibration_align_gui` program may be used::

  flydra_analysis_calibration_align_gui DATAFILE3D.h5 --stim-xml=STIMULUS.xml

This results in a GUI that looks a bit like

.. image:: screenshots/flydra_analysis_calibration_align_gui.png

Using the controls on the right, align your data such that it
corresponds with the 3D model loaded by STIMULUS.xml. When you are
satisfied, click either of the save buttons to save your newly-aligned
calibration.