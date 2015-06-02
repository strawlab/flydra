.. _orientation_smoothing:

Smoothing 3D orientations
=========================

Given the raw Pluecker coordinates estimating body direction found
with :command:`flydra_analysis_orientation_ekf_fitter` (and described
:ref:`here <orientation_ekf_fitter-fusing-2d-orientations-to-3d>`), we
remove the 180 degree ambiguity ("choose orientations") and perform
final smoothing in the code in the module
:mod:`flydra.a2.core_analysis`. Command-line programs that export this
data include :command:`flydra_analysis_data2smoothed` and
:command:`flydra_analysis_montage_ufmfs`. The command-line arguments
to these programs support changing all the various parameters in this
smoothing and orientation selection. Specifically, these parameters are::

  --min-ori-quality-required=MIN_ORI_QUALITY_REQUIRED
                        minimum orientation quality required to emit 3D
                        orientation info
  --ori-quality-smooth-len=ORI_QUALITY_SMOOTH_LEN
                        smoothing length of trajectory
