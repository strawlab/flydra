.. _orientation_smoothing:

Smoothing 3D orientations
=========================

Given the raw Pluecker coordinates estimating body direction found
with :command:`flydra_analysis_orientation_ekf_fitter` (and described
in `<orientation_ekf_fitter-fusing-2d-orientations-to-3d>`_), we
remove the 180 degree ambiguity ("choose orientations") and perform
final smoothing in the code in the module
:mod:`flydra.a2.core_analysis`. One command-line program that exports
this data is :command:`flydra_analysis_data2smoothed`. The
command-line arguments to this program support changing all the
various parameters in this smoothing and orientation selection.
