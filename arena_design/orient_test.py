#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from matplotlib.matlab import *
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg

import VisionEgg.ThreeDeeMath as tdm
from minimization import leastsq # my own code

def render_aa_ellipse(a,b,theta,resolution,save=False):
    dpi=resolution
    figure(figsize=(1,1),dpi=dpi)
    clf()
    
    M = tdm.TransformMatrix()
    M.rotate(theta, 0,0,-1)
    
    ax=axes((0,0,1,1),axisbg='k')

    t = arange(0,2*pi,2*pi/1000.)

    x = a*cos(t)
    y = b*sin(t)
    z = zeros(t.shape)
    w = ones(t.shape)

    verts = array([x,y,z,w])
    v2 = matrixmultiply(M.matrix,verts)

    v2 = v2/v2[3,:]

    x2 = v2[0,:]
    y2 = v2[1,:]

    poly = Polygon( zip(x2,y2), facecolor='w', edgecolor='w')
    ax.add_patch(poly)

    set(ax,'xlim',(-5,5))
    set(ax,'ylim',(-5,5))
    
    set(ax,'xticks',[])
    set(ax,'yticks',[])

    if save:
        fname = 'ellipse%03d.png'%theta
        print 'saving',fname
        savefig(fname,dpi=dpi)

    canvas = get_current_fig_manager().canvas
    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s = agg.tostring_rgb()

    # get the width and the height to resize the matrix
    l,b,w,h = agg.figure.bbox.get_bounds()
    w, h = int(w), int(h)

    X = fromstring(s, UInt8)
    X.shape = h, w, 3
    X = X[:,:,0] # only take red channel to make grayscale

    return X
    

a = 2.0
b = 1.0
resolution = 10
for theta in arange(0.0,180.0,5.0):
    X=render_aa_ellipse(a,b,theta,resolution,save=True)
    
    print X
    
