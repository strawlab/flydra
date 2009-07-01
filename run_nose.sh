#!/bin/bash

# abort on error
set -e

VIRTUALENVDIR=PYtest

# clean old virtual environment
rm -rf $VIRTUALENVDIR

# build new virutal environment
virtualenv $VIRTUALENVDIR

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

# Compile and then install into virtual environment
python setup.py develop

# run X server
Xvfb :2 &
# get PID
XVFBPID=$!
echo "Xvfb running in process $XVFBPID"

# Run tests, capture exit code, don't quit on error.
(DISPLAY=":2" python -c "import nose; nose.main('flydra')";
RESULT=$?)

# Kill the X server
kill $XVFBPID

# exit with test results
exit $RESULT
