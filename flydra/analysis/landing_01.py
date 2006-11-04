import result_browser
import numpy
import pylab

post_diameter = 10 # mm

post = [( 466.8, 191.6, 15.8),# bottom
        ( 467.6, 212.7, 223.4)] # top

def main(max_err=10.0):
    results = result_browser.get_results('DATA20060502_211811.h5',mode='r')
    #data3d = results.root.data3d_fast
    f,xyz,L,err = result_browser.get_f_xyz_L_err(results,max_err=max_err)
    post_top = numpy.array(post[1])
    if 0:
        print xyz.shape
        print post_top.shape
        print '(xyz-post_top)**2',((xyz-post_top)**2).shape
        print 'numpy.sum( (xyz-post_top)**2, axis=1 )',(numpy.sum( (xyz-post_top)**2, axis=1 )).shape
        print 'numpy.sqrt(numpy.sum( (xyz-post_top)**2, axis=1 ))',numpy.sqrt(numpy.sum( (xyz-post_top)**2, axis=1 )).shape
    dist_from_post_top = numpy.sqrt(numpy.sum( (xyz-post_top)**2, axis=1 ))
    pylab.plot(f,dist_from_post_top,'.')
    pylab.show()

if __name__=='__main__':
    main()
