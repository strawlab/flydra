import glob, os, sys
import matplotlib
matplotlib.use('GTKAgg') # TkAgg doesn't work, at least without ioff(), which I haven't tried
import pylab
import tables
import flydra.reconstruct
import flydra.undistort
import flydra.geom
import numpy
from optparse import OptionParser
import flydra.a2.stim_plugins as stim_plugins

nan = numpy.nan
plugin_loader = stim_plugins.PluginLoader()

def my_interp(X0, X1, n):
    dist = X1-X0
    rs=numpy.linspace(0,1,n)
    res = []
    for r in rs:
        res.append( X0 + (r*dist) )
    return res

def doit(filename=None,
         reconstructor_filename=None,
         meters = True,
         draw_stim_func_str = None,
         ):

    PluginClass = plugin_loader(draw_stim_func_str)
    plugin = PluginClass(filename=filename)
    verts, lines = plugin.get_lines()

    all_pts = []

    for poly in lines:
        vert0 = None

        # XXX this is from when I thought poly could have any length.
        # Now I now is will only be len()==2 and this could be simplified.
        poly_verts = [ verts[i] for i in poly ]

        pts = []
        for i in range(1,len(poly_verts)):
            X0 = numpy.array(poly_verts[i-1])
            X1 = numpy.array(poly_verts[i])
            pts.extend( my_interp( X0, X1, 100 ) )
        pts = numpy.array(pts)
        all_pts.append( pts )

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

    for cam_id in cam_ids:
        pylab.title(cam_id)
        distorted = False
        if distorted:
            im = images[cam_id]
        else:
            im = flydra.undistort.undistort(recon,images[cam_id],cam_id)
        pylab.imshow(im,origin='lower')

        for pts in all_pts:
            #print pts
            pt2d = recon.find2d(cam_id, pts, distorted=distorted)

            pylab.plot( pt2d[0,:], pt2d[1,:] )

        ax = pylab.gca()
        ax.set_xlim( (0, images[cam_id].shape[1]) )
        ax.set_ylim( (0, images[cam_id].shape[0]) )

        pylab.show()

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)

    parser.add_option("-r", "--reconstructor", dest="reconstructor_path", type='string',
                      help="calibration/reconstructor path (if not specified, defaults to FILE)",
                      metavar="RECONSTRUCTOR")

    parser.add_option("--draw-stim",
                      type="string",
                      dest="draw_stim_func_str",
                      default=None,
                      help="possible values: %s"%str(plugin_loader.all_names),
                      )

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
         draw_stim_func_str = options.draw_stim_func_str,
         )

if __name__=='__main__':
    main()
