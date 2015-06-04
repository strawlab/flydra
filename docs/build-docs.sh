#!/bin/bash

ARG1=$1

# abort on error
set -o errexit

VIRTUALENVDIR=PYtest

virtualenv --system-site-packages ${VIRTUALENVDIR}

# activate new virutal environment
source $VIRTUALENVDIR/bin/activate

# need to install our own sphinx into virtualenv so it picks up flydra
easy_install docutils-0.8.1.tar.gz
easy_install Jinja2-2.1.1.tar.gz
easy_install Sphinx-1.1.2.tar.gz

cd flydra-sphinx-docs/

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

chmod -R a+r build
find build -type d | xargs chmod -R a+x build

if [ "$ARG1" == "upload" ]; then
    echo "uploading"
    ./rsync-it.sh
fi
