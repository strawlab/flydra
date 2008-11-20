from __future__ import division
import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import cgtypes # import cgkit 1.x
import numpy as np

D2R = np.pi/180.0

def align_pmat( M, P ):
    P = np.dot(P,np.dual.inv(M))
    return P

def cgmat2np(cgkit_mat):
    """convert cgkit matrix to numpy matrix"""
    arr = np.array(cgkit_mat.toList())
    if len(arr)==9:
        arr.shape = 3,3
    elif len(arr)==16:
        arr.shape = 4,4
    else:
        raise ValueError('unknown shape')
    return arr.T

def test_cgmat2mp():
    point1 = (1,0,0)
    point1_out = (0,1,0)

    cg_quat = cgtypes.quat().fromAngleAxis( 90.0*D2R, (0,0,1))
    cg_in = cgtypes.vec3(point1)

    m_cg = cg_quat.toMat3()
    cg_out = m_cg*cg_in
    cg_out_tup = (cg_out[0],cg_out[1],cg_out[2])
    assert np.allclose( cg_out_tup, point1_out)

    m_np = cgmat2np(m_cg)
    np_out = np.dot( m_np, point1 )
    assert np.allclose( np_out, point1_out)

class Alignment(traits.HasTraits):
    s = traits.Float(1.0)
    tx = traits.Float(0)
    ty = traits.Float(0)
    tz = traits.Float(0)

    r_x = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)
    r_y = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)
    r_z = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)

    traits_view = View( Group( ( Item('s'),
                                 Item('tx'),
                                 Item('ty'),
                                 Item('tz'),
                                 Item('r_x',style='custom'),
                                 Item('r_y',style='custom'),
                                 Item('r_z',style='custom'),
                                 )),
                        title = 'Alignment Parameters',
                        )

    def get_matrix(self):
        qx = cgtypes.quat().fromAngleAxis( self.r_x*D2R, cgtypes.vec3(1,0,0))
        qy = cgtypes.quat().fromAngleAxis( self.r_y*D2R, cgtypes.vec3(0,1,0))
        qz = cgtypes.quat().fromAngleAxis( self.r_z*D2R, cgtypes.vec3(0,0,1))
        Rx = cgmat2np(qx.toMat3())
        Ry = cgmat2np(qy.toMat3())
        Rz = cgmat2np(qz.toMat3())
        R = np.dot(Rx, np.dot(Ry,Rz))

        t = np.array([self.tx, self.ty, self.tz],np.float)
        s = self.s

        T = np.zeros((4,4),dtype=np.float)
        T[:3,:3] = s*R
        T[:3,3] = t
        T[3,3]=1.0

        return T
