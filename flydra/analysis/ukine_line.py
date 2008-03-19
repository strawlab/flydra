import glob, os, sys
import matplotlib
matplotlib.use('GTKAgg') # TkAgg doesn't work, at least without ioff(), which I haven't tried
import pylab
import tables
import flydra.reconstruct
import flydra.geom
import numpy
from optparse import OptionParser
nan = numpy.nan
import flydra.undistort

def doit(filename=None,
         reconstructor_filename=None,
         dotype=False,
         meters = True,
         options = None,
         ):

    assert dotype in ['line','point','ray']

    results = tables.openFile(filename,mode='r')

    if reconstructor_filename is None:
        reconst_orig_units = flydra.reconstruct.Reconstructor(results)
    else:
        if reconstructor_filename.endswith('h5'):
            fd = tables.openFile(reconstructor_filename,mode='r')
            reconst_orig_units = flydra.reconstruct.Reconstructor(fd)
        else:
            reconst_orig_units = flydra.reconstruct.Reconstructor(reconstructor_filename)

    if meters:
        recon = reconst_orig_units.get_scaled(reconst_orig_units.get_scale_factor())
    else:
        recon = reconst_orig_units

    image_table = results.root.images
    images = {}
    for row in results.root.cam_info:
        cam_id = row['cam_id']

        arr = getattr(image_table,cam_id)
        image = arr.read()
        images[cam_id] = image
    results.close()
    del results

    cam_ids = images.keys()
    cam_ids.sort()

    class ClickGetter:
        def __init__(self):
            self.coords = []
        def on_click(self,event):
            # get the x and y coords, flip y from top to bottom
            x, y = event.x, event.y
            if event.button==1:
                if event.inaxes is not None:
                    print >> sys.stderr, 'data coords', event.xdata, event.ydata
                    self.coords.append( (event.xdata, event.ydata) )
                    if dotype=='line':
                        if len(self.coords)>2:
                            del self.coords[0]
                    elif dotype=='point':
                        if len(self.coords)>1:
                            del self.coords[0]
                    elif dotype=='ray':
                        if len(self.coords)>1:
                            del self.coords[0]

    click_locations = []

    for cam_id in cam_ids:
        print >> sys.stderr, cam_id

        if dotype=='line':
            title_str = 'line %s: click 2x for line'
        elif dotype=='point':
            title_str = 'point %s: click 1x for point'
        elif dotype=='ray':
            title_str = 'ray %s: click 1x for point'

        pylab.title(title_str%cam_id)
        if options.undistorted_images:
            im = flydra.undistort.undistort(recon,images[cam_id],cam_id)
        else:
            im = images[cam_id]
        pylab.imshow(im,origin='lower')

        cg = ClickGetter()
        binding_id=pylab.connect('button_press_event', cg.on_click)
        pylab.show()
        pylab.disconnect(binding_id)

        if dotype=='point' and len(cg.coords) == 1:
            if options.undistorted_images:
                x = cg.coords[0]
            else:
                x = recon.undistort(cam_id,cg.coords[0])
            click_locations.append( (cam_id,x) )

        if dotype=='ray' and len(cg.coords) == 1:
            if options.undistorted_images:
                x0 = cg.coords[0]
            else:
                x0 = recon.undistort(cam_id,cg.coords[0])
            ray1 = recon.get_projected_line_from_2d(cam_id,x0)
            print 'ray=',repr(ray1)

        if dotype=='line' and len(cg.coords) == 2:
            # user clicked 2 (or more) times

            # find 2d coordinates of clicked points
            if options.undistorted_images:
                x0 = cg.coords[0]
                x1 = cg.coords[1]
            else:
                x0 = recon.undistort(cam_id,cg.coords[0])
                x1 = recon.undistort(cam_id,cg.coords[1])

            # find 3d points on ray from camera through clicked points
            X0=numpy.dot( recon.pmat_inv[cam_id], [x0[0],x0[1],1.0] )
            X1=numpy.dot( recon.pmat_inv[cam_id], [x1[0],x1[1],1.0] )
            # camera center is 3rd 3d point
            C = flydra.reconstruct.pmat2cam_center( recon.get_pmat(cam_id) )
            C = list(C.flat)+[1.0]
            A = numpy.array( [X0, X1, C] )
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

    if dotype=='line':
        line3d = recon.find3d( click_locations,
                               return_X_coords = False,
                               return_line_coords = True )
        print 'line3d=',repr(line3d)
    elif dotype=='point':
        X = recon.find3d( click_locations, return_line_coords = False )

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
        print 'X=(%s, %s, %s)'%(repr(X[0]),repr(X[1]),repr(X[2]))

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)

    parser.add_option("-r", "--reconstructor", dest="reconstructor_path", type='string',
                      help="calibration/reconstructor path (if not specified, defaults to FILE)",
                      metavar="RECONSTRUCTOR")

    parser.add_option("--dotype", default='point')

    parser.add_option("--force-distorted-images",
                      dest='undistorted_images',
                      action='store_false',
                      default=True )

    (options, args) = parser.parse_args()

    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return

    if len(args)<1:
        parser.print_help()
        return

    h5_filename=args[0]
    doit(filename = h5_filename,
         reconstructor_filename=options.reconstructor_path,
         dotype=options.dotype,
         options = options,
         )

if __name__=='__main__':
    main()
