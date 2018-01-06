#!/bin/bash -x
set -e

# abort on error
set -o errexit

VIRTUALENVDIR=PYtest

virtualenv --system-site-packages ${VIRTUALENVDIR}

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

pip install Sphinx==1.6.5

cd flydra-sphinx-docs/

echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"

# clean
rm -rf build

# # make
# make latex
# cd build/latex
# make all-pdf
# cd ../..
make html

chmod -R a+r build
find build -type d | xargs chmod -R a+x build
