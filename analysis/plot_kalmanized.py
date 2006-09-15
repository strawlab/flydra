import tables as PT
import numpy
import pylab

filename='DATA20060830_184701.tracked_fixed_accel.h5'
results = PT.openFile(filename, mode="r")
obj_ids = results.root.kalman_estimates.read(field='obj_id',flavor='numpy')

minx = 0.0
maxx = 0.6

miny = 0.0
maxy = 0.3

xbins = numpy.linspace(minx,maxx,200)
ybins = numpy.linspace(miny,maxy,100)

for this_obj_id in range(obj_ids.max()+1):
    print this_obj_id,'of',obj_ids.max()
    rc = numpy.nonzero( obj_ids == this_obj_id )[0]
    x = results.root.kalman_estimates.readCoordinates(rc,field='x',flavor='numpy')
    y = results.root.kalman_estimates.readCoordinates(rc,field='y',flavor='numpy')
    pylab.plot(x,y)
    
pylab.show()
    
