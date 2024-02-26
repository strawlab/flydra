import os
import sys
from flydra import *
import warnings

__version__ = "0.1.0"

err_str = ('The module "flydra" is deprecated. Use "flydra_core" instead.')
if os.environ.get('FLYDRA_NO_BACKWARDS_COMPAT','0') != '0':
    raise ImportError(err_str)
warnings.warn(err_str, DeprecationWarning)
