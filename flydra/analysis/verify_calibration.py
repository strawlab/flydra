from flydra.reconstruct import *
from pylab import *
ioff()
import numarray as nx

def get_camera_order(filename):
    fd = open(filename,'r')
    cam_ids = fd.read().split('\n')
    fd.close()
    if cam_ids[-1] == '': del cam_ids[-1] # remove blank line
    return cam_ids

IdMat = load_ascii_matrix('IdMat.dat')
points = load_ascii_matrix('points.dat')

cam_order = get_camera_order('camera_order.txt')

R = Reconstructor('.')

# find all 3d points possible
try:
    X3d
except NameError:
    X3d = {}
    print 'finding 3d...'
    for j in xrange(IdMat.shape[1]):
        this_col = IdMat[:,j]
        if sum(this_col) < 2: # not capable of finding 3d
            continue
        d2={}
        for i, cam_id in enumerate(cam_order):
            if this_col[i]:
                xd = (points[ 3*i,j ], points[ 3*i+1,j ]) # distorted coords
                d2[cam_id] = R.undistort(cam_id,xd) # save linear coords
        X = R.find3d(d2.items())
        X3d[j] = X
    print 'done'

draw_plots = False
cam_err = {}
for i, cam_id in enumerate(cam_order):
    print 'calculating',cam_id
    this_cam_2d_orig = []
    this_cam_2d_repr = []
    js = X3d.keys()
    js.sort()
    #js = [ j for j in js if j%10==0 ]
    cam_err[cam_id]=0.0
    for j in js:
        if IdMat[i,j]:
            xd = (points[ 3*i,j ], points[ 3*i+1,j ]) # distorted
            xl = R.undistort(cam_id,xd) # make linear
            xdtest = R.distort(cam_id,xl)
            assert (xd[0]-xdtest[0])**2 < 1e10
            assert (xd[1]-xdtest[1])**2 < 1e10
            this_cam_2d_orig.append( xl )
            X = X3d[j]
            xr = R.find2d(cam_id,X)
            this_cam_2d_repr.append(xr[:2])

            dist = math.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2)
            cam_err[cam_id] += dist/len(js)
    print '    ',cam_id,cam_err[cam_id]
    if draw_plots:
        this_cam_2d_orig = nx.array(this_cam_2d_orig)
        this_cam_2d_repr = nx.array(this_cam_2d_repr)
    
        figure(i+1)
        lines=plot(this_cam_2d_orig[:,0],this_cam_2d_orig[:,1],'o')
        #set(lines,'markerfacecolor',None)
        set(lines,'markerfacecolor',None)
        set(lines,'markeredgecolor',(0,1,0)) # green = orig
        lines=plot(this_cam_2d_repr[:,0],this_cam_2d_repr[:,1],'o')
        set(lines,'markerfacecolor',None)
        set(lines,'markeredgecolor',(0,0,1)) # blue = repr
        grid()
    print 'done'
    #break

print 'mean err',sum(cam_err.values())/len(cam_err)
if draw_plots:
    show()
