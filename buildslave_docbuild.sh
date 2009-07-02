#!/bin/bash

ARG1=$1

# abort on error
set -o errexit

VIRTUALENVDIR=PYtest

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

# need to install our own sphinx into virtualenv so it picks up flydra
easy_install Jinja2-2.1.1.tar.gz
easy_install Sphinx-0.6.2.tar.gz

cd flydra-sphinx-docs/
./get-svn.sh

echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"

# clean
rm -rf build

# make
make latex
cd build/latex
make all-pdf
cd ../..
make html

if [ "$ARG1" == "upload" ]; then
    echo "uploading"
    rsync-it.sh
fi
