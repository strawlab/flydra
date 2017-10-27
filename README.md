# flydra - multi-camera tracking system

Flydra is a markerless, multi-camera tracking system capable of tracking the
three-dimensional position and body orientation of animals such as flies and
birds. The system operates with less than 40 ms latency and can track multiple
animals simultaneously. Fundamentally, the multi-target tracking algorithm is
based on an extended Kalman filter and the nearest neighbour standard filter
data association algorithm.

## Discussion

For questions or discussion, please use [the "multicams" Google
Group](https://groups.google.com/forum/#!forum/multicams).

## Installation

For installation, we recommend using [our Ansible
playbooks](https://github.com/strawlab/strawlab-ansible-roles.git). In particular,
the `ros-kinetic-flydra` role or the `ros-kinetic-freemovr` install on
Ubuntu 16.04 with ROS Kinetic, either flydra alone or within a [full FreemoVR
system](https://strawlab.org/freemovr).

## History

This software was originally develped by Andrew Straw in the Dickinson Lab at
Caltech from 2004-2010. Ongoing development continued, coordinated by the Straw
Lab, from 2010. The software was open sourced in 2017.

## Publications

Flydra is described in the following papers:

Straw AD✎, Branson K, Neumann TR, Dickinson MH. Multicamera Realtime 3D Tracking
of Multiple Flying Animals. *Journal of The Royal Society Interface* 8(11),
395-409 (2011)
[doi:10.1098/rsif.2010.0230](https://dx.doi.org/10.1098/rsif.2010.0230).

Stowers JR*, Hofbauer M*, Bastien R, Griessner J⁑, Higgins P⁑, Farooqui S⁑,
Fischer RM, Nowikovsky K, Haubensak W, Couzin ID, Tessmar-Raible K✎, Straw AD✎.
Virtual Reality for Freely Moving Animals. Nature Methods (2017)
[doi:10.1038/nmeth.4399](https://dx.doi.org/10.1038/nmeth.4399).

Please cite these if you use Flydra.

## License

With the exception of third party software, the software, documentation and
other resouces are licensed under either of

* Apache License, Version 2.0,
  (./LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (./LICENSE-MIT or http://opensource.org/licenses/MIT)
  at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

## Code of conduct

Anyone who interacts with flydra in any space including but not
limited to this GitHub repository is expected to follow our [code of
conduct](https://github.com/strawlab/flydra/blob/master/code_of_conduct.md).
