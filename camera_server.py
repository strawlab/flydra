#!/usr/bin/env python

try:
    import flydra.camera_server_pyrex as x
except ImportError:
    import flydra.camera_server_noipp_pyrex as x
x.main()

