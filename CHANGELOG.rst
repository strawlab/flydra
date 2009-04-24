Major changes for 0.4.40
------------------------

  * Framecounts saved from trigger device correspond exactly with live
    framecounts. This should allow for automatic synchronization
    detection (which is not yet implemented).

  * Added :command:`flydra_analysis_get_clock_sync` to check the
    synchronization of the mainbrain and camera computer clocks in
    DATAFILE2D.h5 files.

  * Make generation and use of timestamp synchronization files
    (extension .spreadh5) more explicit.

  * :command:`flydra_analysis_plot_kalman_2d`: add --show-orientation
    and --autozoom command line options

Major changes for 0.4.39
------------------------

  * Fix debian/control to depend on python-traitsgui

Major changes for 0.4.38
------------------------

  * Further fixes to data analysis and plotting code to deal with
    un-synchronized cameras

  * flydra_analysis_calibration_align_gui: show stimulus in correct
    location with outline in (bugfix)

  * flydra_analysis_plot_timeseries_2d_3d: speedups when plotting 3D
    data and --start and --stop are given

  * flydra_analysis_plot_timeseries_3d: speedups when --start and
    --stop are given

  * Documenation improvements, particularly calibration section.

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
