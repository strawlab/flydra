#!/bin/bash -x
set -e

rosservice call flydra_mainbrain/do_synchronization
