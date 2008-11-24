Major changes for 0.4.37
------------------------

  * Enable automatic running of most unit tests

  * flydra_kalmanize: bugfix to fix exception when iterating file chunks

Major changes for 0.4.36
------------------------

  * flydra_kalmanize: ensure that cameras were synchronized.

  * flydra_kalmanize: significant speedup when working on giant files
    through chunked reads.

  * flydra_mainbrain: allow sound output when tracking in 3D.

  * flydra_mainbrain: emit target obj_id when tracking (required a
    change in the data packet API and format, see r3050).

  * flydra_analysis_calibration_align_gui: bugfix for stimulus
    position.

  * flydra_analysis_calibration_align_gui: allow flipping of
    individual coordinate axes.

  * Further documentation and testing improvements.
