# flydra change log

## unreleased

### Fixed

* calculate fps even when 'timestamp' column is nan

## [0.7.5] - 2018-08-20

### Changed

* Calculate framerate from data when not explicity specified in saved metadata.
* Work towards Python 3 compatibility

### Fixed

* Better handling of nan values in data.

## [0.7.4] - 2018-05-29

### Added

* Created documentation for calibrating cameras in
  `docs/flydra-sphinx-docs/calibrating.md`

### Fixed

* Updated analysis scripts that use VTK to VTK 6 (the version shipped in
  Ubuntu 16.04).
* Do not load timezone in plot_timeseries_2d_3d unless it is needed.
* Made camnode compatible with "Allied Vision Technologies" cameras by
  renaming them to "AVT". The spaces and length of the name caused problems
  within ROS.

## [0.7.3] - 2018-01-31

### Fixed

* Expand user (`~`) when writing file in flydra_core.

## [0.7.2] - 2018-01-17

### Fixed

* Save local timezone name correctly and compatible with pytz in flydra_core.

## [0.7.1] - 2018-01-17

### Changed

* Fixed packaging issues with .deb files.

## [0.7.0] - 2018-01-17

### Changed

* First major release as open source software.
* Reorganized code into three packages: `flydra_core`, `flydra_analysis` and
  `flydra_camnode`. Removed lots of outdated material and unused and unsupported
  code.
* Reduce the number of threads spawned by the coordinate processor thread in
  mainbrain.
* Realtime priority of coordinate processor is not elevated by default.
* Realtime priority can be set with `posix_scheduler` ROS parameter, e.g. to
  `['FIFO', 99]` to match the previous behavior.

### Added

* Make attempting to recover missed 2D data settable via environment variable
  `ATTEMPT_DATA_RECOVERY`.

### Fixed

* Documentation builds again
* Fixed a regresssion in which saving 2D data was blocked when no 3D calibration
  was loaded. Also correctly close hdf5 file in these circumstances.
* Do not quit mainbrain if a previously seen camera re-joins after suddenly
  quitting.

[0.7.5]: https://github.com/strawlab/flydra/compare/release/0.7.4...release/0.7.5
[0.7.4]: https://github.com/strawlab/flydra/compare/release/0.7.3...release/0.7.4
[0.7.3]: https://github.com/strawlab/flydra/compare/release/0.7.2...release/0.7.3
[0.7.2]: https://github.com/strawlab/flydra/compare/release/0.7.1...release/0.7.2
[0.7.1]: https://github.com/strawlab/flydra/compare/release/0.7.0...release/0.7.1
[0.7.0]: https://github.com/strawlab/flydra/compare/release/0.6.14...release/0.7.0
