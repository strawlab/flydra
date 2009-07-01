#!/bin/bash

# abort on error
set -e

VIRTUALENVDIR=PYtest

# clean old virtual environment
rm -rf $VIRTUALENVDIR

# build new virutal environment
virtualenv $VIRTUALENVDIR

# compile extentions
$VIRTUALENVDIR/bin/python setup.py build_ext --inplace

# install into virtual environment
$VIRTUALENVDIR/bin/python setup.py develop

# run X server
Xvfb :2 &
# get PID
XVFBPID=$!
echo "Xvfb running in process $XVFBPID"

# Run tests, capture exit code, don't quit on error.
(DISPLAY=":2" PATH=$VIRTUALENVDIR/bin $VIRTUALENVDIR/bin/python -c "import nose; nose.main('flydra')"; \
RESULT=$?)

# Kill the X server
kill $XVFBPID

# exit with test results
exit $RESULT
