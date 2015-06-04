#!/bin/bash -x

# Change the revision number below and then run this script to update
# numpy installation.

rm -rf ext
svn co -r 8716 http://svn.scipy.org/svn/numpy/trunk/doc/sphinxext ext
find ext -name '.svn' -print0 | xargs -0 rm -rf
