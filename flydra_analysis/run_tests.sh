#!/bin/bash
set -o errexit
# tests specified in setup.cfg
ETS_TOOLKIT='null' python -c "import matplotlib; matplotlib.use('Agg'); import nose; nose.run_exit()" --eval-attr="not (known_fail or slow_command)" $*
