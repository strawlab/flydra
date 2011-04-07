#!/bin/bash

# abort on error
set -o errexit

rsync -rlpvz --delete-after -P build/html/ xen1:/var/websites/code.astraw.com/flydra-doc/
#rsync -rlpvz -P build/latex/flydra.pdf xen1:/var/websites/code.astraw.com/flydra-doc/
