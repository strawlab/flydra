import os
opj=os.path.join
import numarray as na
import numarray.linear_algebra as la

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    lines = buf.split('\n')[:-1]
    return na.array([map(float,line.split()) for line in lines])

def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def uP2X(Umat,Ps):
    """

    arguments
    ---------
    Umat: 3*N x n matrix of n homogenous points
    Ps:   3 x 4*N matrix of projection matrices

    returns
    -------
    X:    4 x n matrix of homogenous 3D points
    """
    N=Umat.shape[0]/3
    n=Umat.shape[1]
    
    # reshuffle the Ps matrix
    Pmat = []
    for i in range(N):
        Pmat.append( Ps[0,i*4:i*4+4] )
        Pmat.append( Ps[1,i*4:i*4+4] )
        Pmat.append( Ps[2,i*4:i*4+4] )
    Pmat = na.array(Pmat)

    X = []
    for i in range(n): # for all points
        A = []
        for j in range(N): # for all cameras
            # create the data matrix
            A.append( ( Umat[j*3  ,i]*Pmat[j*3+2,:] - Pmat[j*3  ,:]) )
            A.append( ( Umat[j*3+1,i]*Pmat[j*3+2,:] - Pmat[j*3+1,:]) )
        A=na.asarray(A)
        u,s,v=la.singular_value_decomposition(A)
        X.append( v[-1,:] )
    
    #normalize reconstructed points
    X=na.array(X)
    X.transpose()
    w=X[3,:]
    X=X/na.resize(w,(4,X.shape[1]))
    return X

class Reconstructor:
    def __init__(self,
                 CAMS = 4,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 ):
        # load calibration matrices
        self.Pmat=[load_ascii_matrix(opj(calibration_dir,Pmat_name%(i+1))) for i in range(CAMS)]


    def find3d(self,
               cam_idx,
               points2d):

        Wsx = []
        for p in points2d:
            Wsx.extend( [p[0], p[1], 1.0] )

        Wsx = na.array(Wsx)
        Wsx = Wsx[:,na.NewAxis]
        Pmatx = na.concatenate([self.Pmat[i] for i in cam_idx],axis=1)
        X=uP2X(Wsx,Pmatx)
        return X
    
def test():
    r=Reconstructor()
    
    points2d = [ (286, 46.333333),
                 (366, 404.33333),
                 (320.33333, 156.33333),
                 (428, 433.33333), ]
    
    cam_idx = range(4)
    
    pt3d=r.find3d(cam_idx,points2d)
    print pt3d

if __name__=='__main__':
    test()
