#!/bin/bash

# exit on error
set -e

rm -rf build
make latex
cd build/latex
make all-pdf
cd ../..
make html

./rsync-it.sh
