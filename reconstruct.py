import os
opj=os.path.join
#import numarray as nx
#from numarray.linear_algebra import singular_value_decomposition
import Numeric as nx
from LinearAlgebra import singular_value_decomposition

def load_ascii_matrix(filename):
    fd=open(filename,mode='rb')
    buf = fd.read()
    lines = buf.split('\n')[:-1]
    return nx.array([map(float,line.split()) for line in lines])

class Reconstructor:
    def __init__(self,
                 CAMS = 4,
                 calibration_dir = '/home/astraw/mcsc_data',
                 Pmat_name = 'camera%d.Pmat.cal',
                 ):
        # load calibration matrices
        Pmat=[load_ascii_matrix(opj(calibration_dir,Pmat_name%(i+1))) for i in range(CAMS)]
        self.Pmat=nx.concatenate(Pmat,axis=0)

    def find3d(self,
               cam_idx_and_points2d):
        M=len(cam_idx_and_points2d) # number of views of single point
        A=nx.zeros((2*M,4))
        
        for m,(cam,(xm,ym)) in enumerate(cam_idx_and_points2d):
            row2=self.Pmat[m*3+2,:]
            A[2*m  ,:]=xm*row2-self.Pmat[m*3  ,:]
            A[2*m+1,:]=ym*row2-self.Pmat[m*3+1,:]

        u,s,v=singular_value_decomposition(A)
        X = v[-1,0:3]/v[-1,3] # normalize
        return X
    
def test():
    r=Reconstructor()
    points2d = [ (286, 46.333333),
                 (366, 404.33333),
                 (320.33333, 156.33333),
                 (428, 433.33333), ]

    cam_idx = range(4)

    pt3d=r.find3d(zip(cam_idx,points2d))
    print 'pt3d',pt3d

if __name__=='__main__':
    test()
