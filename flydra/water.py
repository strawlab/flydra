import numpy as np
import flydra.refract as refract
R2D = 180./np.pi

class WaterInterface:
    '''for now empty

    If water is ever not at z=0, this will be the place to implement it.
    '''
    def __init__(self):
        # values for air and water from
        # http://en.wikipedia.org/wiki/Refraction
        self.n1 = 1.0003
        self.n2 = 1.3330

def view_points_in_water( reconstructor, cam_id, pts3d, water, distorted=True ):
    assert isinstance(water,WaterInterface)

    pts3d = np.array(pts3d)
    assert pts3d.ndim==2
    assert pts3d.shape[1]==3 # pts3d.shape[1] == npts

    pt1 = reconstructor.get_camera_center(cam_id)[:,0]
    pt2 = np.array(pt1,copy=True); pt2[2]=0 # closest point to camera on water surface, assumes water at z==0

    shifted_pts = pts3d - pt2 # origin under cam at surface. cam at (0,0,z).
    theta = np.arctan2( shifted_pts[:,1], shifted_pts[:,0] ) # angles to points
    r = np.sqrt(np.sum(shifted_pts[:,:2]**2,axis=1)) # horizontal dist
    depth = -shifted_pts[:,2]

    height = pt1[2]

    shifted_water_surface_pts = np.zeros_like( pts3d )

    r0 = []
    for i in range(len(shifted_pts)):
        assert depth[i]>=0
        r0.append(
            refract.fermat1( water.n1, water.n2,
                             height, r[i], depth[i]))

    r0=np.array(r0)
    shifted_water_surface_pts[:,0] = r0*np.cos(theta)
    shifted_water_surface_pts[:,1] = r0*np.sin(theta)

    water_surface_pts = np.ones( (shifted_water_surface_pts.shape[0],4) )
    water_surface_pts[:,:3] = shifted_water_surface_pts + pt2

    pts = reconstructor.find2d( cam_id, water_surface_pts,
                                distorted=distorted,
                                bypass_refraction=True,
                                )
    return pts
