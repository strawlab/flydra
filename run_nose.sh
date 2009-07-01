#!/bin/bash

# abort on error
set -o errexit

VIRTUALENVDIR=PYtest

# clean old virtual environment
rm -rf $VIRTUALENVDIR

# build new virutal environment
virtualenv $VIRTUALENVDIR

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

easy_install nose-0.11.1.tar.gz

# Compile and then install into virtual environment
python setup.py develop


#----- Now, run tests -----

# check for expected permissions on X server
python -c 'import os; assert os.path.exists("/etc/X2.hosts")'
python -c 'fd = open("/etc/X2.hosts",mode="r"); assert "localhost\n" in fd.readlines()'

# run X server
Xvfb :2 &
# get PID
XVFBPID=$!
echo "Xvfb running in process $XVFBPID"

echo "checking that Xvfb process is running"
sleep 1
python -c "import os; assert os.path.exists(\"/proc/$XVFBPID\")"
echo "PID $XVFBPID seems OK, will continue with tests"

# Run tests, capture exit code, don't quit on error.
set +o errexit
#DISPLAY=":2" nosetests --verbosity=3 -A "not slow_command"
DISPLAY=":2" nosetests -A "not known_fail"
#DISPLAY=":2" nosetests --verbosity=3
RESULT=$?
set -o errexit

# Kill the X server
kill $XVFBPID
wait $XVFBPID

# exit with test results
exit $RESULT
