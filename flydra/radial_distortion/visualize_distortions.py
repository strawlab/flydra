import numpy
import pylab
import flydra.reconstruct_utils as reconstruct_utils
from optparse import OptionParser

def info(msg):
    if 0:
        print msg

def visualize_distortions( ax, helper, width=None, height=None ):
    if (width is None) or (height is None):
        K = helper.get_K()
        x0 = K[0,2]
        y0 = K[1,2]
        if height is None:
            height = y0*2
        if width is None:
            width = x0*2
        info('guessing image width and/or height from principal point')

    aspect = width/height
    nwide = 50
    nhigh = int(nwide/aspect)
    x = numpy.linspace(0,width,nwide)
    y = numpy.linspace(0,height,nhigh)

    X,Y = numpy.meshgrid(x,y)
    shape = X.shape

    Xu = numpy.nan*numpy.ones(shape)
    Yu = numpy.nan*numpy.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            Xu[i,j], Yu[i,j] = helper.undistort( X[i,j], Y[i,j] )

    U = Xu-X
    V = Yu-Y
    dsi = slice(0,shape[0],3)
    dsj = slice(0,shape[1],3)
    ax.quiver( X[dsi,dsj],Y[dsi,dsj], U[dsi,dsj],V[dsi,dsj], units='x', scale=1.0)

    shift_mag = numpy.sqrt(U**2 + V**2)
    CS = pylab.contour(X,Y,shift_mag)
    pylab.clabel(CS, inline=1, fontsize=10)

    K = helper.get_K()
    x0 = K[0,2]
    y0 = K[1,2]
    ax.plot([x0],[y0],'ko')

def main():
    parser = OptionParser(usage='%prog RAD_FILE',
                          version="%prog 0.1")
    (cli_options, args) = parser.parse_args()
    if not len(args)==1:
        raise RuntimeError('one command-line argument is needed - the .rad file')
    radfile = args[0]
    helper = reconstruct_utils.make_ReconstructHelper_from_rad_file(radfile)
    fig = pylab.figure()
    ax = fig.add_subplot(1,1,1)
    visualize_distortions( ax, helper )
    pylab.show()

if __name__=='__main__':
    main()
