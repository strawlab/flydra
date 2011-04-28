.. _orientation_smoothing:

Smoothing orientations with flydra
==================================

Given Pluecker coordinates estimating body direction found with
<orientation_ekf_fitter-fusing-2d-orientations-to-3d>, we remove the
180 degree ambiguity ("choose orientations") and perform final
smoothing in the code in the :module:`flydra.a2.core_analysis`. This
code is meant to be called directly from Python to perform the task at
one. One command-line program that exports the data is
:command:`flydra_analysis_data2smoothed`.
