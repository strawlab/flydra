##import numarray as nx
import Numeric as nx
import matplotlib
import matplotlib.matlab as mpl
import matplotlib.cm
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.mlab import meshgrid
from wxPython.wx import *

origin = 'upper'

class PlotPanel(wxPanel):

    def __init__(self, parent):
        wxPanel.__init__(self, parent, -1)

        self.fig = mpl.Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas) #matplotlib toolbar
        self.toolbar.Realize()
        #self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wxBoxSizer(wxVERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wxLEFT|wxTOP|wxGROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wxGROW)
        self.SetSizer(sizer)
        self.Fit()

    def init_plot_data(self):
        a = self.fig.add_axes([0.05,0.05,0.9,0.9])

        start_size=656,491
        x = nx.arange(start_size[0])
        y = nx.arange(start_size[1])
##        mpl.set(a,'xlim',[0,start_size[0]])
##        mpl.set(a,'xticks',range(0,start_size[0],100))
##        mpl.set(a,'ylim',[0,start_size[1]])
##        mpl.set(a,'yticks',range(0,start_size[1],100))
        x, y = meshgrid(x, y)
        z = nx.zeros(x.shape)
        self.im = a.imshow( z,
                            cmap=matplotlib.cm.jet,
                            origin=origin,
                            interpolation='nearest')
        #self.im.set_clim(0,255)
        
        self.lines = a.plot([0],[0],'o-') 	 
        #self.lines = a.plot([0,0,0],[0,0,0],'o-') 	 
        mpl.set(self.lines[0],'markerfacecolor',None) 	 
        white = (1.0,1.0,1.0) 	 
        mpl.set(self.lines[0],'color',white) 	 
        mpl.set(self.lines[0],'linewidth',2.0) 	 
        mpl.set(self.lines[0],'markeredgecolor',white) 	 
        mpl.set(self.lines[0],'markeredgewidth',2)
        a.grid('on')
        self.toolbar.update() # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an 
        # unmanaged toolbar in your frame
        return self.toolbar

    def set_image(self,image):
        orig_shape = self.im.get_size()
        if image.shape[0] != orig_shape[0] or image.shape[1] != orig_shape[1]:
            print "main_brain WARNING: size changed to %s, don't know how to re-adjust"%str(image.shape)
        self.im.set_array(image)
        
    def set_points(self,points):
        zp = zip(*points)
        #self.lines[0].set_data(zp[0],zp[1])
        if origin == 'upper':
            y = 490-nx.asarray(zp[1])
        else:
            y = zp[1]
        self.lines[0].set_data(zp[0],y)

    def draw(self):
        self.canvas.draw()
		
    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

