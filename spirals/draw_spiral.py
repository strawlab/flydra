from matplotlib.matlab import *
from numarray import *

figure(figsize=(8,8),frameon=False)
axes((0.0,0.0,1.0,1.0))
set(gca(),'frame_on',False)
theta = arange(0,8*pi,0.1)
a=1
b=.2

for dt in arange(0,2*pi,pi/2.0):
    
    x = a*cos( theta+dt )*exp(b*theta)
    y = a*sin( theta+dt )*exp(b*theta)

    dt = dt+pi/4.0
    
    x2 = a*cos( theta+dt )*exp(b*theta)
    y2 = a*sin( theta+dt )*exp(b*theta)

    xf = concatenate( (x,x2[::-1]) )
    yf = concatenate( (y,y2[::-1]) )

    p1=fill(xf,yf)
    set(p1[0],'facecolor',(0.0,1.0,0.0))
    set(p1[0],'linewidth',0)
    
#show()
savefig('spiral.png')
