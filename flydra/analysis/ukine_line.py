import glob, os, sys
import FlyMovieFormat
import pylab
import tables
import flydra.reconstruct
import numpy
nan = numpy.nan

# base file names
base_fname = 'full_20060516_191746_%s_bg.fmf'
# hdf5 file containing calibration data
cal_source = 'newDATA20060516_194920.h5'

cams = ['cam%d'%i for i in range(1,6)]

h5file = tables.openFile(cal_source,mode='r')
recon = flydra.reconstruct.Reconstructor(h5file)
h5file.close()
del h5file

class ClickGetter:
    def __init__(self):
        self.coords = []
    def on_click(self,event):
        # get the x and y coords, flip y from top to bottom
        x, y = event.x, event.y
        if event.button==1:
            if event.inaxes is not None:
                print >> sys.stderr, 'data coords (distorted)', event.xdata, event.ydata
                self.coords.append( (event.xdata, event.ydata) )
                if len(self.coords)>2:
                    del self.coords[0]

click_locations = []
for cam in cams:
    cam_id = None
    for c in recon.cam_ids:
        if c.startswith(cam):
            if cam_id is not None:
                raise RuntimeError('>1 camera per host not yet supported')
            cam_id = c
    fname = base_fname%cam_id
    print >> sys.stderr, cam_id,fname
    
    fmf = FlyMovieFormat.FlyMovie(fname)
    frame,timestamp = fmf.get_frame(0)
    fmf.close()
    
    pylab.imshow(frame,origin='lower')

    cg = ClickGetter()
    binding_id=pylab.connect('button_press_event', cg.on_click)
    pylab.show()
    pylab.disconnect(binding_id)
    if len(cg.coords) == 2:
        # user clicked 2 (or more) times

        # find 2d coordinates of clicked points
        x0 = recon.undistort(cam_id,cg.coords[0])
        x1 = recon.undistort(cam_id,cg.coords[1])

        # find 3d points on ray from camera through clicked points
        X0=numpy.dot( recon.pmat_inv[cam_id], [x0[0],x0[1],1.0] )
        print 'X0',X0
        X1=numpy.dot( recon.pmat_inv[cam_id], [x1[0],x1[1],1.0] )
        print 'X1',X1
        # camera center is 3rd 3d point
        C = flydra.reconstruct.pmat2cam_center( recon.get_pmat(cam_id) )
        C = list(C.flat)+[1.0]
        print 'C', C
        A = numpy.array( [X0, X1, C] )
        print 'A',A
        u,d,vt = numpy.linalg.svd(A,full_matrices=True)
        Pt = vt[3,:] # plane parameters
        p1,p2,p3,p4 = Pt[0:4]

        x=nan
        y=nan
        area=nan
        slope=nan
        eccentricity = 1e10
        value_tuple = x,y,area,slope,eccentricity, p1,p2,p3,p4
        click_locations.append( (cam_id,value_tuple) )
        
    print >> sys.stderr

line3d = recon.find3d( click_locations,
                       return_X_coords = False,
                       return_line_coords = True )
print 'line3d=',repr(line3d)

##l2norms = []
##for cam_id,orig_2d_undistorted in click_locations:
##    predicted_2d_undistorted = recon.find2d( cam_id, X )
##    o = numpy.asarray(orig_2d_undistorted)
##    p = numpy.asarray(predicted_2d_undistorted)
##    l2norm = numpy.sqrt(numpy.sum((o-p)**2))
##    print >> sys.stderr, '%s (% 5.1f, % 5.1f) (% 5.1f,% 5.1f) % 5.1f'%(cam_id,
##                                                        orig_2d_undistorted[0],
##                                                        orig_2d_undistorted[1],
##                                                        predicted_2d_undistorted[0],
##                                                        predicted_2d_undistorted[1],
##                                                        l2norm)
##    l2norms.append( l2norm )

##h5file.close()
##print >> sys.stderr
##print >> sys.stderr, 'mean reconstruction error:',numpy.mean(l2norms)
##print >> sys.stderr
##print 'X=(% 5.1f,% 5.1f,% 5.1f) # 3d location'%(X[0],X[1],X[2])
