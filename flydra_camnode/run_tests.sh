#!/bin/bash
set -o errexit
# tests specified in setup.cfg
ETS_TOOLKIT='null' python -c "import nose; nose.run_exit()" --eval-attr="not (known_fail or slow_command)" $*
