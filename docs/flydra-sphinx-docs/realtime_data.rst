Realtime data
=============

Flydra is capable of outputing a low latency stream of state estimate
from the Kalman filter.

The data are originally encoded into a "data packet" by
:meth:`flydra_core.kalman.flydra_tracker.Tracker.encode_data_packet`.

Then, they are transferred to a separate thread in the
:class:`flydra_core.MainBrain.CoordinateSender` class. There, multiple
packets are combined into a "super packet" with
:func:`flydra_core.kalman.data_packets.encode_super_packet`. By combining
into a single buffer to send over the network, fewer system calls are
made, resulting in better performance and reducing overall latency.

