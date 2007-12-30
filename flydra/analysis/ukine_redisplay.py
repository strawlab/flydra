import glob, os, sys
import flydra.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
import pylab
import tables
import flydra.reconstruct
import numpy
import numarray
import flydra.undistort as undistort

# last used 2006-05-19

# base file names
base_fname = 'full_20060516_191746_%s_bg.fmf'
# hdf5 file containing calibration data
cal_source = 'newDATA20060516_194920.h5'
verts = [( 443.9, 247.1,  7.8),
         ( 456.9, 243.2, 226.7)
         ]
linesegs = [[0,1]]

# Pluecker coordinates
lines = [(-0.42616909392791952, -0.9033477045229068, -0.045123578401282435,
          0.017456206233934717, -0.0012089233634138012, 0.0007142508901631997)]

cams = ['cam%d'%i for i in range(1,6)]

h5file = tables.openFile(cal_source,mode='r')
recon = flydra.reconstruct.Reconstructor(h5file)
h5file.close()

if 1:
    for cam in cams:
        cam_id = None
        for c in recon.cam_ids:
            if c.startswith(cam):
                if cam_id is not None:
                    raise RuntimeError('>1 camera per host not yet supported')
                cam_id = c
        fname = base_fname%cam_id

        fmf = FlyMovieFormat.FlyMovie(fname)
        frame,timestamp = fmf.get_frame(0)
        fmf.close()

        pylab.figure()
        pylab.title( cam_id )
        # undistort image
        undistorted = flydra.undistort.undistort(recon,frame,cam_id)
        pylab.imshow(undistorted,origin='lower')
        verts2d = [ recon.find2d(cam_id,X,distorted=False) for X in verts ]
        for lineseg in linesegs:
            x = [ verts2d[i][0] for i in lineseg ]
            y = [ verts2d[i][1] for i in lineseg ]
            pylab.plot( x, y, 'w-' )

pylab.show()
