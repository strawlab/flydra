import scipy.misc
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    pix_diff_threshold=5.0

    im1=scipy.misc.pilutil.imread(sys.argv[1])
    im2=scipy.misc.pilutil.imread(sys.argv[2])

    di = abs(im1.astype(np.float)-im2.astype(np.float) )
    if di.ndim==3:
        # flatten
        di2d = np.mean(di,axis=2)
    else:
        di2d = di

    n_diff = np.sum(di2d > pix_diff_threshold)
    n_total = di2d.shape[0]* di2d.shape[1]
    fraction_different = n_diff/float(n_total)
    fraction_same = 1.0-fraction_different

    print 'fraction_same=%s'%fraction_same
    plt.imshow(di2d,origin='upper',interpolation='nearest')
    plt.colorbar()
    plt.show()

if __name__=='__main__':
    main()
