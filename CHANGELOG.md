# flydra change log

## unreleased

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

[0.7.1]: https://github.com/strawlab/flydra/compare/release/0.7.0...release/0.7.1
[0.7.0]: https://github.com/strawlab/flydra/compare/release/0.6.14...release/0.7.0
