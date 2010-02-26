#!/bin/bash -x
set -e
find www -name '*~' | xargs rm -f
rsync -avzP www/ flydra-webapp:www/
