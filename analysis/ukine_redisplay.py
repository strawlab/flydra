import glob, os, sys
import FlyMovieFormat
import pylab
import tables
import flydra.reconstruct
import numpy
import numarray
import flydra.undistort as undistort

# base file names
base_fname = 'landing_20060502_full_bg.fmf'
# hdf5 file containing calibration data
cal_source = 'DATA20060502_211811.h5'

verts = [( 466.8, 191.6, 15.8),# bottom
        ( 467.6, 212.7, 223.4)] # top
linesegs = [[0,1]]

# Pluecker coordinates
lines = [(0.094779836509674989, 0.99467657888647865, 0.022775459998087867,
          -0.033385095420257797, 0.00087061695304787039, -0.0011143866774977608)]
intersect_line_planes = [('yz',466)]

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
        fname = os.path.join(cam,base_fname)

        fmf = FlyMovieFormat.FlyMovie(fname)
        frame,timestamp = fmf.get_frame(0)
        fmf.close()

        pylab.figure()
        pylab.title( cam_id )
        if 0:
            pylab.imshow(frame,origin='lower')
            verts2d = [ recon.find2d(cam_id,X,distorted=True) for X in verts ]
            for lineseg in linesegs:
                x = [ verts2d[i][0] for i in lineseg ]
                y = [ verts2d[i][1] for i in lineseg ]
                pylab.plot( x, y, 'w-' )
        else:
            # undistort image
            if 0:
                undistorted = flydra.undistort.undistort(recon,frame,cam_id)
            else:
                #naa = numarray.asarray
                #naa = numpy.asarray
                intrin = recon.get_intrinsic_linear(cam_id)
                k = recon.get_intrinsic_nonlinear(cam_id)
                f = intrin[0,0], intrin[1,1] # focal length
                c = intrin[0,2], intrin[1,2] # camera center
                im = undistort.rect(frame, f=f, c=c, k=k)
                im = numpy.asarray(im)
                undistorted = im.astype(numpy.UInt8)
            pylab.imshow(undistorted,origin='lower')
            verts2d = [ recon.find2d(cam_id,X,distorted=False) for X in verts ]
            for lineseg in linesegs:
                x = [ verts2d[i][0] for i in lineseg ]
                y = [ verts2d[i][1] for i in lineseg ]
                pylab.plot( x, y, 'w-' )
        
pylab.show()
