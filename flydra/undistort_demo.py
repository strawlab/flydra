import tables
import numpy
import scipy
from optparse import OptionParser
import scipy.misc.pilutil
import sys
import flydra.reconstruct as reconstruct
import flydra.undistort as undistort
import pylab

if 1:
    filename = sys.argv[1]
    results = tables.openFile(filename,mode='r')
    R = reconstruct.Reconstructor(results)

    x = numpy.linspace( 100, 400, 25 )
    y = numpy.linspace( 100, 400, 25 )

    for row in results.root.cam_info:
        cam_id = row['cam_id']

        helper = R.get_reconstruct_helper_dict()[cam_id]
        print 'creating mesh...',cam_id
        lbrt = -100, -100, 750, 600
        dm = undistort.DistortionMesh( helper, lbrt, xdim=400, ydim=400 )
        print 'created mesh'

        coll = []
        for xi in x:
            this_line = []
            print 'undistorting xi',xi
            xi = xi*numpy.ones_like(y)
            undistorted_x, undistorted_y = dm.undistort_points(  xi, y )
            coll.append( (undistorted_x, undistorted_y) )

        for yi in y:
            this_line = []
            print 'undistorting yi',yi
            yi = yi*numpy.ones_like(x)
            undistorted_x, undistorted_y = dm.undistort_points(  x, yi )
            coll.append( (undistorted_x, undistorted_y) )

        if 1:
            pylab.figure()
            for (undistorted_x, undistorted_y) in coll:
                pylab.plot( undistorted_x, undistorted_y, 'b-' )
            pylab.title('backprojection of grid through lens %s'%cam_id)
            ax = pylab.gca()
            ax.set_xlim( (0, 500))
            ax.set_ylim( (0, 500))
            #break
    pylab.show()

if 0:
    filename = sys.argv[1]
    results = tables.openFile(filename,mode='r')
    R = reconstruct.Reconstructor(results)

    image_table = results.root.images

    shown = False
    for row in results.root.cam_info:
        cam_id = row['cam_id']

        arr = getattr(image_table,cam_id)
        image = arr.read()

        helper = R.get_reconstruct_helper_dict()[cam_id]

        lbrt = 0, 0, 650, 500
        print 'creating mesh...'
        dm = undistort.DistortionMesh( helper, lbrt )
        print 'created mesh'
        undistorted = dm.undistort_image( image )

        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.imshow( image )

        pylab.subplot(2,1,2)
        ucond = ~numpy.isnan( undistorted )
        uim = numpy.zeros_like( undistorted )
        uim[ ucond ] = undistorted[ucond]
        pylab.imshow( undistorted )

        shown = True
        break

    if shown:
        pylab.show()

