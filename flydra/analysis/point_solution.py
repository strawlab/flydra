#!/usr/bin/env python

from flydra.reconstruct import Reconstructor, load_ascii_matrix
import os
import numarray as nx
opj = os.path.join

# p. 323 Hartley & Zizzerman
# p. 61 Press 

cal_dir = '/home/astraw/Cal-2004-11-08'
R = Reconstructor( calibration_dir = cal_dir, debug=True )

IdMat = load_ascii_matrix( opj( cal_dir, 'IdMat.dat' ) )
points = load_ascii_matrix( opj( cal_dir, 'points.dat' ) )

col = 100
cam_idx = IdMat[:,100]
N = len(cam_idx)
pts_i = points[:,100]
pts_i = nx.reshape( pts_i, (N,3) )
arg = []
for cam_id, use_it, pt in zip(R.cam_order, cam_idx, pts_i):
    if use_it:
        arg.append( (cam_id, (pt[0], pt[1]) ) )
X = R.find3d( arg )
print 'X',X

