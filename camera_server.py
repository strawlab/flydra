#!/usr/bin/env python
import sys

try:
    import camera_server_pyrex as x
    print sys.argv[0],'using IPP'
except ImportError, x:
    print 'loading IPP failed'
    import traceback
    traceback.print_exc()
    import camera_server_noipp_pyrex as x
    print sys.argv[0],'running (not using IPP)'
x.main()

