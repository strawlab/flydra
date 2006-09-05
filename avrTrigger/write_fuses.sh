#!/bin/bash
uisp -dprog=stk500 -dserial=/dev/ttyS0 -dpart=ATMEGA169 --wr_fuse_l=0xcf
