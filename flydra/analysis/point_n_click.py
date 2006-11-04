import glob, os, sys
import matplotlib
matplotlib.use('GTKAgg') # TkAgg doesn't work, at least without ioff(), which I haven't tried
import pylab
import numpy
import Image

class ClickGetter:
    def on_click(self,event):
        # get the x and y coords, flip y from top to bottom
        x, y = event.x, event.y
        if event.button==1:
            if event.inaxes is not None:
                print >> sys.stderr, 'data coords (distorted)', event.xdata, event.ydata
                self.coords = event.xdata, event.ydata

click_locations = []
if 1:
    fname = sys.argv[1]
    print >> sys.stderr,fname
    im = Image.open(fname)
    imdata = numpy.fromstring(im.tostring('raw','L',0,-1),dtype=numpy.UInt8)
    imdata.shape = im.size[1], im.size[0]
    pylab.imshow(imdata,origin='lower')

    cg = ClickGetter()
    binding_id=pylab.connect('button_press_event', cg.on_click)
    pylab.show()
    pylab.disconnect(binding_id)


