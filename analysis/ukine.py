import glob, os, sys
import FlyMovieFormat
import pylab
import tables
import flydra.reconstruct
import numpy

# base file names
base_fname = 'full_20060516_191746_%s_bg.fmf'
# hdf5 file containing calibration data
cal_source = 'newDATA20060516_194920.h5'

h5file = tables.openFile(cal_source,mode='r')
recon = flydra.reconstruct.Reconstructor(h5file)
h5file.close()
del h5file

class ClickGetter:
    def on_click(self,event):
        # get the x and y coords, flip y from top to bottom
        x, y = event.x, event.y
        if event.button==1:
            if event.inaxes is not None:
                print >> sys.stderr, 'data coords (distorted)', event.xdata, event.ydata
                self.coords = event.xdata, event.ydata

click_locations = []
for cam_id in recon.cam_ids:
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
    if hasattr(cg,'coords'):
        # user clicked
        click_locations.append( (cam_id, recon.undistort(cam_id,cg.coords) ))
    print >> sys.stderr

X = recon.find3d( click_locations )

l2norms = []
for cam_id,orig_2d_undistorted in click_locations:
    predicted_2d_undistorted = recon.find2d( cam_id, X )
    o = numpy.asarray(orig_2d_undistorted)
    p = numpy.asarray(predicted_2d_undistorted)
    l2norm = numpy.sqrt(numpy.sum((o-p)**2))
    print >> sys.stderr, '%s (% 5.1f, % 5.1f) (% 5.1f,% 5.1f) % 5.1f'%(cam_id,
                                                        orig_2d_undistorted[0],
                                                        orig_2d_undistorted[1],
                                                        predicted_2d_undistorted[0],
                                                        predicted_2d_undistorted[1],
                                                        l2norm)
    l2norms.append( l2norm )

print >> sys.stderr
print >> sys.stderr, 'mean reconstruction error:',numpy.mean(l2norms)
print >> sys.stderr
print 'X=(% 5.1f,% 5.1f,% 5.1f) # 3d location'%(X[0],X[1],X[2])
