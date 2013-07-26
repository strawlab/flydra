#!/usr/bin/env python

# This is a backwards compatibility file to allow Cython module in
# global namespace to continue to work.
import os

no_backwards_compat = int(os.environ.get('FLYDRA_NO_BACKWARDS_COMPAT','0'))
if no_backwards_compat:
    raise RuntimeError('you are using an outdated import')

from _pmat_jacobian import *
