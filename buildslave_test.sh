#!/bin/bash

# abort on error
set -o errexit

VIRTUALENVDIR=PYtest

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

# install nose tests
easy_install nose-0.11.1.tar.gz

#----- Now, run tests -----

# check for expected permissions on X server
python -c 'import os; assert os.path.exists(os.path.expanduser("~/.xvfb-display")), "no ~/.xvfb-display"'
python -c 'import os; d=int(open(os.path.expanduser("~/.xvfb-display")).read()[1:]); assert os.path.exists("/etc/X%d.hosts"%d), "no /etc/X%d.hosts file exists for xvfb %d"'
python -c 'import os; d=int(open(os.path.expanduser("~/.xvfb-display")).read()[1:]); fd = open("/etc/X%d.hosts"%d,mode="r"); assert "localhost\n" in fd.readlines(), "no localhost in /etc/X%d.hosts"%d'

# run X server
export DISPLAY=`cat ~/.xvfb-display`
Xvfb $DISPLAY &
# get PID
XVFBPID=$!
echo "Xvfb running in process $XVFBPID"

echo "checking that Xvfb process is running"
sleep 1
python -c "import os; assert os.path.exists(\"/proc/$XVFBPID\")"
echo "PID $XVFBPID seems OK, will continue with tests"

# Run tests, capture exit code, don't quit on error.
set +o errexit
#nosetests --verbosity=3 -A "not slow_command"
nosetests -A "not known_fail" --all-modules
#nosetests --verbosity=3
RESULT=$?
set -o errexit

# Kill the X server
kill $XVFBPID
wait $XVFBPID

# exit with test results
exit $RESULT
