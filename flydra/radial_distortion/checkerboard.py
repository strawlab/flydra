from __future__ import division
import sys,time,os,pprint

# import the necessary things for OpenCV
import CVtypes
from CVtypes import cv
import ctypes, math
import numpy
import numpy as np
import pylab
import scipy.optimize
import scipy.misc.pilutil
import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
import motmot.imops.imops as imops
from matplotlib import delaunay
import flydra.reconstruct_utils as reconstruct_utils # in pyrex/C for speed
import flydra.undistort
from flydra.reconstruct import angles_near
import scipy.cluster.vq
#import networkx as NX
import simplenx as NX

from optparse import OptionParser
from visualize_distortions import visualize_distortions

D2R = math.pi/180.0
R2D = 180.0/math.pi

def info(msg):
    if 0:
        print msg

def get_color(i):
    colors = ['r','g','b','y','c']
    color=colors[i%len(colors)]
    return color

def get_singly_connected_nodes(graph):
    result = []
    for node in graph.nodes():
        if len(graph.neighbors(node)) == 1:
            result.append( node )
    return result

def find_subgraph_similar_direction(G,
                                    source=None,
                                    direction_eps_radians=None,
                                    already_done=None,
                                    ):
    """
    originally from networkx/search.py

    added traverse_node_callback() stuff.
    """

    neighbors=G.neighbors

    seen={}   # nodes seen
    succ={}

    queue=[source]     # use as LIFO queue
    direction_radians = None
    already_did_first_edge = False
    debug = False
    while queue:
        v=queue[-1]
        if v not in seen:
            seen[v]=True
            succ[v]=[]
        done=1
        for w in neighbors(v):
            if w not in seen:

                this_direction_radians = w.get_direction_from( v )
                if direction_radians is None:
                    # testing first edge

                    for test_graph in already_done:
                        # check already done graphs to see if we have this edge
                        already_did_first_edge = test_graph.has_edge( v,w )
                        if already_did_first_edge:
                            break

                    # first edge
                    if not already_did_first_edge:
                        direction_radians = this_direction_radians

                if ((not already_did_first_edge) and
                    angles_near(this_direction_radians,direction_radians,direction_eps_radians, mod_pi=True)):
                    queue.append(w)
                    succ[v].append(w)
                    done=0
                    break
                else:
                    seen[w] = True
        if done==1:
            queue.pop()
    result = None
    if len( succ ) > 1:
        result = NX.Graph(succ)
        if 0:
            print 'source',source
            print succ
            print
    return result

class CornerNode:
    def __init__(self,x,y,name,aspect_ratio=1.0):
        self._x=float(x)
        self._y=float(y)
        self._r = {}
        self._name=int(name)
        self._aspect_ratio=float(aspect_ratio)
    def __repr__(self):
        return 'CornerNode(%s,%s,%s,aspect_ratio=%s)'%(repr(self._x),
                                                       repr(self._y),
                                                       repr(self._name),
                                                       repr(self._aspect_ratio))
    def __hash__(self):
        return self._name
    def __cmp__(self,other):
        if isinstance(other,CornerNode):
            return self._name.__cmp__(other._name)
        else:
            raise ValueError('cannot compare CornerNode against anything but '
                             'a CornerNode')
    def __str__(self):
        return str(self._name)
    def get_pos(self):
        return (self._x, self._y)
    def get_rand_pos(self,g):
        """return a slightly shifted point position for unique key g"""
        if g not in self._r:
            rx = self._x + 2*np.random.normal(size=(1,))
            ry = self._y + 2*np.random.normal(size=(1,))
            self._r[g] = (rx,ry)
        return self._r[g]
    def get_direction_from( self, v ):
        """return direction of self from v in radians"""
        x1, y1 = v.get_pos()
        x2, y2 = self.get_pos()

        yd = y2-y1
        xd = x2-x1
        xd *= self._aspect_ratio
        mag = math.sqrt(xd**2 + yd**2)
        return math.atan2(yd/mag, xd/mag)
    def get_distance_from( self, v ):
        """return distance of self from v"""
        x1, y1 = v.get_pos()
        x2, y2 = self.get_pos()

        yd = y2-y1
        xd = x2-x1
        xd *= self._aspect_ratio
        mag = math.sqrt(xd**2 + yd**2)
        return mag

def get_direction_stats(g,mod_pi=True):
    vecs = []
    for n0,n1 in g.edges():
        theta = n0.get_direction_from(n1)
        x = np.cos(theta)
        y = np.sin(theta)
        if mod_pi and y<0:
            y = -y
            x = -x
        vecs.append( (x,y) )
    vecs = np.array(vecs)
    #print vecs
    mx,my = np.mean(vecs,axis=0)
    theta_r = np.sqrt(mx**2+my**2)
    theta_mean = np.arctan2(my,mx)
    thetas = np.arctan2(vecs[:,1],vecs[:,0])
    theta_median = np.median(thetas)
    stats = {'mean':theta_mean,
             'median':theta_median,
             'r':theta_r,
             'thetas':thetas}
    return stats

def points2graph(x,y,
                 distance_thresh=1.5,
                 angle_thresh=30*D2R,
                 show_clusters=False,
                 show_clusters_frame=None,
                 aspect_ratio = 1.0,
                 ):
    x = numpy.array(x)
    y = numpy.array(y)
    tri = delaunay.Triangulation(x, y)
    nodes = [ CornerNode(xi,yi,i,aspect_ratio=aspect_ratio) for i,(xi,yi) in enumerate(zip(x,y)) ]

    segx = []
    segy = []
    vert_inds = []
    for node in tri.triangle_nodes:
        for i in range(3):
            segx.append( ( x[node[i]], x[node[(i+1)%3] ] )  )
            segy.append( ( y[node[i]], y[node[(i+1)%3] ] )  )
            vert_inds.append( ( node[i], node[(i+1)%3] ) )

        if 0:
            # find and remove hypotenuse
            dist2 = [ (segx[i][0] - segx[i][1])**2 + (segy[i][0] - segy[i][1])**2
                      for i in range(-3, 0) ]
            longest_ind = -3 + numpy.argmax( dist2 )
            del segx[longest_ind]
            del segy[longest_ind]
            del vert_inds[longest_ind]

    # Discard duplicates.  (This could be acheived by more careful
    # attention at the triangulation stage.)
    idx = 0
    while idx < len(vert_inds):
        test = vert_inds[idx]
        idx += 1

        remove = []
        for cmpi in range(idx,len(vert_inds)):
            if test == vert_inds[cmpi]:
                remove.append( cmpi )
        remove.reverse()
        for i in remove:
            del vert_inds[i]
            del segx[i]
            del segy[i]

    adjacency = numpy.zeros( ( len(x), len(x) ), dtype=numpy.uint32 )
    graph = NX.Graph()
    for test_seg in vert_inds:
        i,j = test_seg
        graph.add_edge( nodes[i], nodes[j] )
        # The graph is not directed, so we don't need to add (j,i).

    if 1:
        # remove edges not belonging to 2 shortest distance clusters
        edges = graph.edges()

        directions = [edge[0].get_direction_from( edge[1] ) for edge in edges]
        directions = numpy.array(directions)%numpy.pi
        distance   = [edge[0].get_distance_from( edge[1] ) for edge in edges]

        obs = numpy.array([directions,distance]).T
        distance_max = obs[:,1].max()
        scale = [[1,numpy.pi/distance_max]]
        scaled_obs = obs*scale

        if 1:
            # do clustering on Cartesian grid.

            r = obs[:,1]
            median_r = numpy.median(r)
            x = r*numpy.cos( obs[:,0]*2 ) # double angle to go around full circle
            y = r*numpy.sin( obs[:,0]*2 ) # double angle to go around full circle
            if 0 and show_clusters:
                pylab.figure()
                pylab.plot(x,y,'.')
                ax = pylab.gca()
                ax.set_aspect('equal')

                print 'median_r',median_r
                print '%d cut'%len(numpy.nonzero(r > median_r*distance_thresh )[0])

            # threshold large distances to origin
            good_cond = r <= median_r*distance_thresh
            good_idx = numpy.nonzero(good_cond)[0]

            cartesian_obs = numpy.array([x,y]).T
            # filter data
            cartesian_obs_use = cartesian_obs[good_idx]

            # 4 clusters: 2 for each main direction, 2 for diagonals
            #cartesian_clusters, labels = scipy.cluster.vq.kmeans2( cartesian_obs, 4)
            # 5 clusters: same as above, plus trash
            if 0:
                cartesian_clusters, labels = scipy.cluster.vq.kmeans2( cartesian_obs, 5, iter=100, minit='points')
            else:
                # initial guesses
                cluster_guesses = numpy.array([[median_r,0],
                                               [0,median_r],
                                               [-median_r,0],
                                               [0, -median_r]])
                cartesian_clusters_use, labels_use = scipy.cluster.vq.kmeans2( cartesian_obs_use,
                                                                               cluster_guesses,
                                                                               iter=100,
                                                                               minit='matrix',
                                                                               )
                # add new cluster with filtered data
                new_label = numpy.max( labels_use ) + 1
                cartesian_clusters = numpy.zeros( (5,2) )
                cartesian_clusters[:-1,:] = cartesian_clusters_use
                labels = new_label*numpy.ones( (cartesian_obs.shape[0],), dtype = labels_use.dtype)
                labels[good_idx] = labels_use
                cartesian_clusters_center = numpy.array(cartesian_clusters,copy=True)
                ## print 'cartesian_clusters'
                ## print cartesian_clusters
                ## print
            x = numpy.array(cartesian_clusters[:,0],copy=True)
            y = numpy.array(cartesian_clusters[:,1],copy=True)
            r = numpy.sqrt(x**2 + y**2)
            theta = numpy.arccos( y/r, x/r )
            theta[ r==0 ] = 0 # eliminate nan
            theta = theta/2 # get back to mod pi angles
            clusters = numpy.array([ theta, r ]).T

        if show_clusters:
            n_clusters = len(clusters)
            pylab.figure()

            ax = pylab.subplot(2,1,1)
            for i in range(n_clusters):
                cluster_cond = labels==i
                this_obs = obs[cluster_cond]
                color=get_color(i)
                pylab.plot( this_obs[:,0]/2.0, this_obs[:,1],'.',mec=color,mfc=color )
                #pylab.plot([clusters[i][0]],[clusters[i][1]],'ko')
                print 'label',i,[clusters[i][0]],[clusters[i][1]]
            if show_clusters_frame is not None:
                pylab.title('frame %d'%show_clusters_frame)

            ax=pylab.subplot(2,1,2)
            for i in range(n_clusters):
                cluster_cond = labels==i
                this_obs = obs[cluster_cond]
                color=get_color(i)
                pylab.plot( cartesian_obs[cluster_cond,0], cartesian_obs[cluster_cond,1],'.',mec=color,mfc=color )
                #print i,[cartesian_clusters_center[i][0]],[cartesian_clusters_center[i][1]]
                pylab.plot([cartesian_clusters_center[i][0]],[cartesian_clusters_center[i][1]],'ko')
            ax.set_aspect('equal')
            pylab.show()
            #sys.exit()

        cluster_distances = clusters[:,1]
        if show_clusters:
            print 'cluster_distances',cluster_distances
        cluster_idxs = numpy.argsort( cluster_distances )
        shortest_idxs = cluster_idxs[1:3] # shortest is trash at 0, ignore it and take 2 near shortest

        graph = NX.Graph() # new graph
        for i in shortest_idxs:
            take_edges = numpy.nonzero(labels == i)[0]
            this_cluster_directions = obs[take_edges][:,0]
            this_cluster_distances = obs[take_edges][:,1]
            median_cluster_direction = numpy.median(this_cluster_directions)
            median_cluster_distance = numpy.median(this_cluster_distances)
            ## print 'mean',numpy.mean(this_cluster_distances)
            ## print 'median',numpy.median(this_cluster_distances)
            ## print 'std',numpy.std(this_cluster_distances)
            ## print
            for j in take_edges:
                this_distance = obs[j,1]
                this_direction = obs[j,0]
                if this_distance >= distance_thresh*median_cluster_distance:
                    # too long - ignore
                    continue

                if not angles_near(this_direction,median_cluster_direction,angle_thresh,mod_pi=True):
                    # angle is too different
                    continue
                graph.add_edge( edges[j] )

    return graph, nodes

def fit_line( xys ):
    x = xys[:,0]
    y = xys[:,1]
    A = numpy.ones( (x.shape[0], 2) )
    A[:,0] = x
    x, residues, rank, s = numpy.linalg.lstsq(A,y)
    return x, residues, rank, s

## def get_helper_for_params( params ):
##     1/0

## def correct_image( im, params ):
##     helper = get_helper_for_params( params )
##     return helper.undistort_image( im )

class Objective:
    def __init__(self, graphs,
                 width=640, height=480,
                 debug=False,
                 save_debug_images=False,
                 aspect_ratio=1.0,
                 ):
        self._debug = debug
        self._graphs = graphs
        self._xys = []
        self._aspect_ratio = aspect_ratio
        self._width = width
        self._height = height
        self._save_debug_images = save_debug_images
        self._last_err_time = time.time()
        for graph in self._graphs:
            periphery = get_singly_connected_nodes( graph )
            start_node = min(periphery) # ensure this is deteriministic
            ordered = NX.search.dfs_preorder( graph, source=start_node )
            xys = numpy.array([ node.get_pos() for node in ordered ])
            self._xys.append( xys )

    def get_default_p0(self,config=None):
        """create initial estimate of parameter vector"""
	if config is None:
            config={}
        # K13 and K23 may be None in dict, thus default value in config.get() won't work.
        x0 = config.get('K13')
        y0 = config.get('K23')
        if x0 is None:
            x0 = self._width/2.0
        if y0 is None:
            y0 = self._height/2.0
        r1 = config.get( 'kc1', 0.0)
        r2 = config.get( 'kc2', 0.0)

        params = [x0, y0, r1, r2]

        for orig_xys in self._xys:
            line_params = self._fit_line( orig_xys )
            params.extend( line_params )

        return numpy.array(params,dtype=numpy.float64)

    def get_helper_for_params(self,params):
        x0, y0, r1, r2 = params[:4]

        fc0=1000.0
        fc1=fc0*self._aspect_ratio
        tangential1 = tangential2 = 0.0

        helper = reconstruct_utils.ReconstructHelper( fc0, fc1, x0, y0, r1, r2,
                                                      tangential1, tangential2)
        if 0:
            class ReverseHelper:
                def __init__(self,h):
                    self.h = h
                def undistort(self, *args):
                    return self.h.distort(*args)
                def get_K(self):
                    return self.h.get_K()
                def undistort_image(self,*args,**kw):
                    print 'ERROR: image is wrong!'
                    return self.h.undistort_image(*args,**kw)
            rh = ReverseHelper(helper)
            return rh
        else:
            return helper

    def lm_err_func(self, params):
        results = self.lm_err4(params)
        now = time.time()
        if self._debug or (now-self._last_err_time) >= 5.0:
            print 'With these parameters:',repr(params[:4])
            print '  current error is:',numpy.sum( results**2 )
            self._last_err_time = now

        if self._save_debug_images:
            if not hasattr(self,'_save_count'):
                self._save_count = 1
                pylab.figure()
            pylab.clf()
            ax = pylab.gca()
            self.plot_fit( ax, params )
            pylab.savefig( 'debug%03d.png'%self._save_count )
            #print 'saving'
            self._save_count += 1


        if 1:
            # add penalty for straying too far from center
            alpha = 1.0
            helper = self.get_helper_for_params( params )
            K = helper.get_K()
            x0 = K[0,2]
            y0 = K[1,2]
            x0_guess = self._width/2.0
            y0_guess = self._height/2.0
            dist2 = numpy.sqrt((x0-x0_guess)**2 + (y0-y0_guess)**2)
            results = list(results) + [alpha*dist2]
        return results

    def sumsq_err(self, params):
        results = self.lm_err4(params)
        results = numpy.sum( results**2 )
        return results

    def _fit_line( self, xys ):
        #XXX TODO: directly fit line rather than this crazy NL leastsq operation

        def residuals( line_params, xys ):
            d = self._get_dist_from_line( line_params, xys )
            return d

        p0 = numpy.ones((2,))
        pfinal, ier = scipy.optimize.leastsq( residuals, p0,
                                              args=( xys, ),
                                              )
        if ier not in (1,2,3,4):
            raise RuntimeError('could not fit line')
        theta, dist = pfinal

        return theta, dist

    def _get_dist_from_line( self, line_params, xys ):
        theta, dist = line_params
        # point coordinates
        x0 = xys[:,0]
        y0 = xys[:,1]

        return (x0*numpy.cos( theta ) + y0*numpy.sin( theta ) - dist)

    def plot_fit( self, ax, params ):
        helper = self.get_helper_for_params( params )

        for i,orig_xys in enumerate(self._xys):
            new_xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
            ax.plot( [ox for (ox,oy) in orig_xys ],
                        [oy for (ox,oy) in orig_xys ], 'r.' )
            ax.plot( [ox for (ox,oy) in new_xys ],
                        [oy for (ox,oy) in new_xys ], 'bo', mec='b', mfc='None')

            line_params = params[4+i*2:4+(i+1)*2]
            theta, dist = line_params

            # this is way sub-optimal, but...
            if 1:
                # good for near horizontal lines
                xi = numpy.linspace(0,(self._width-1),5)
                yi = (dist-xi*numpy.cos(theta))/numpy.sin(theta)
                valid_cond = (yi > 0) & (yi < self._height)
                nvalid = numpy.sum(valid_cond)
                if nvalid:
                    ax.plot( xi[valid_cond], yi[valid_cond], 'b-', lw=2 )

            if 1:
                # good for near vertical lines
                yi = numpy.linspace(0,(self._height-1),5)
                xi = (dist-yi*numpy.sin(theta))/numpy.cos(theta)
                valid_cond = (xi > 0) & (xi < self._width)
                nvalid = numpy.sum(valid_cond)
                if nvalid:
                    ax.plot( xi[valid_cond], yi[valid_cond], 'b-', lw=2 )

        K = helper.get_K()
        x0 = K[0,2]
        y0 = K[1,2]
        pylab.plot([x0],[y0],'ko')

        pstr = ' '.join(['%.3g'%f for f in params[:4]])

        ax.set_title( '%.3g   %s'%(self.sumsq_err(params), pstr))


    def lm_err4(self, params):
        """

        This was implemented after 'Line-Based Correction of Radial
        Lens Distortion' (GMIP 1997) by Prescott and McLean.

        Also, as decribed in 'Correcting Radial Distortion by Circle
        Fitting' (BMVC 2005) by Rickard Strand and Eric Hayman, this
        algorithm seems to more-or-less called 'DF' (after F. Devernay
        and O.D. Faugeras. Straight lines have to be straight. MVA,
        2001). (I have not read the DF paper, however.)

        Note that the Strand and Hayman paper (see above) suggests
        what might be a better way to estimate radial distortion and
        gives test on synthetic data regarding the performance of
        various algorithms.

        Finally, 'A new algorithm to correct fish-eye and strong
        wide-angle lens-distortion from single images' (2001) by
        Brauer-Burchardt, C.; Voss, K 10.1109/ICIP.2001.958994 gives
        an algorithm that seems very similar to Strand and Hayman.

        """
        helper = self.get_helper_for_params( params )

        results = []
        for i,orig_xys in enumerate(self._xys):
            # each set of xys should form a line after undistortion
            new_xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
            line_params = params[4+i*2:4+(i+1)*2]
            d = self._get_dist_from_line( line_params, new_xys )
            results.extend( list(d) )
        results = numpy.array(results)
        return results

def get_non_background( im, bg, eps=10 ):
    """return the pixels different than background"""
    assert len( im.shape )==2
    absdiff = abs( numpy.asarray(im).astype(numpy.float32) -
                   numpy.asarray(bg).astype(numpy.float32))
    take_cond = absdiff > eps
    newim = numpy.zeros( im.shape, dtype=im.dtype )
    newim[take_cond] = im[take_cond]
    return newim

def binarize( im ):
    median = numpy.median( im )
    newim = numpy.where( im > median, numpy.uint8(255), numpy.uint8(0) )
    return newim

def extract_corners(imnx_use,max_ncorn_per_side=30):
    im_ptr = cv.CreateImage( cv.Size( imnx_use.shape[1], imnx_use.shape[0] ),
                             CVtypes.IPL_DEPTH_8U, 1 )
    ctypes.memmove( im_ptr.contents.imageData,
                    imnx_use.ctypes.data,
                    imnx_use.shape[0]*imnx_use.shape[1] )

    ncorn = max_ncorn_per_side,max_ncorn_per_side
    ncorn_tot = ncorn[0]* ncorn[1]
    corners = (cv.Point2D32f * ncorn_tot)()
    corner_count = ctypes.c_int(ncorn_tot)

    sz = cv.Size(*ncorn)

    flags = 0
    cv.FindChessboardCorners( im_ptr, sz,
                              ctypes.byref(corners[0]),
                              ctypes.byref(corner_count),
                              flags )
    if 0:
        im_ptr = cv.CreateImage( cv.Size( imnx_orig.shape[1], imnx_orig.shape[0] ),
                             CVtypes.IPL_DEPTH_8U, 1 )
        # this seems not so robust and perhaps not so critical with lots of points
        cv.FindCornerSubPix( im_ptr, ctypes.byref(corners[0]),
                             corner_count, sz, cv.Size(1,1),
                             cv.TermCriteria(cv.TERMCRIT_EPS|cv.TERMCRIT_ITER,
                                             10,3))
    else:
        info('not doing sub-pixel corner location')

    cv.ReleaseImage( im_ptr )

    x = []
    y = []
    for i in range( corner_count.value ):
        x.append( corners[i].x )
        y.append( corners[i].y )
    x = np.array(x)
    y = np.array(y)
    return x,y

def test_extract_corners():
    dirname = os.path.split(__file__)[0]
    fname = 'distorted.fmf'
    fullpath = os.path.join(dirname,fname)
    fmf = FlyMovieFormat.FlyMovie(fullpath)
    im,timestamp = fmf.get_frame(0)
    imnx_rawbinary = binarize(im)
    imnx_use = imnx_rawbinary
    actual_x,actual_y=extract_corners(imnx_use)

    if 0:
        pylab.imshow(imnx_use)
        pylab.plot(actual_x,actual_y,'o')
        pylab.title('found corners')
        pylab.show()

    actual = np.array( np.hstack( (actual_x[:,np.newaxis],
                                   actual_y[:,np.newaxis])))

    expected_x= np.array([
        305.5,  261.5,  351.5,  348.5,  302. ,  256.5,  214. ,  218.5,
        298.5,  253. ,  209.5,  169. ,  173.5,  250. ,  207.5,  132. ,
        167. ,  130.5,  135.5,  141. ,  206.5,  167. ,  130. ,  166.5,
        129. ,  206. ,  206. ,  167. ,  130.5,  248.5,  248. ,  247. ,
        292.5,  291. ,  293.5,  340.5,  338.5,  336.5,  386.5,  383.5,
        389. ,  438. ,  434.5,  431.5,  483. ,  479.5,  486. ,  534. ,
        530.5,  525.5,  576. ,  569.5,  579.5,  623. ,  618. ,  613. ,
        659. ,  652.5,  663.5,  701. ,  625. ,  665.5,  703. ,  737. ,
        735.5,  666. ,  703. ,  626. ,  666. ,  625.5,  703.5,  736.5,
        665.5,  703. ,  737. ,  735.5,  664. ,  626. ,  583.5,  583.5,
        539.5,  538.5,  582.5,  537.5,  493.5,  492. ,  581.5,  490.5,
        536. ,  489. ,  444.5,  442.5,  440.5,  394. ,  391.5,  396.5,
        346.5,  343. ,  295.5,  398.5,  446. ])

    expected_y = np.array([
        19. ,   20. ,   19.5,   42. ,   41.5,   42. ,   42. ,   20.5,
         64.5,   64. ,   64.5,   64. ,   42.5,   87.5,   86.5,   64.5,
         86.5,   86.5,   42.5,   22.5,  109.5,  108.5,  107.5,  130.5,
        129. ,  132. ,  154.5,  152. ,  150.5,  111. ,  134. ,  157. ,
        135.5,  158.5,  112.5,  113.5,  137.5,  161. ,  139. ,  162.5,
        114.5,  116.5,  140.5,  164. ,  142. ,  165.5,  117.5,  119. ,
        143. ,  167. ,  145.5,  167. ,  119.5,  122. ,  145.5,  168.5,
        146. ,  168. ,  123.5,  125. ,   98. ,  100.5,  102.5,  103.5,
        125.5,   77.5,   80. ,   53. ,   55. ,   75.5,   58. ,   61. ,
         34. ,   37. ,   40. ,   20. ,   13. ,   30.5,   27.5,   50. ,
         25. ,   48. ,   73. ,   71. ,   23. ,   45.5,   96.5,   69. ,
         95. ,   93.5,   44. ,   67.5,   91.5,   67. ,   90.5,   42.5,
         65.5,   88. ,   88.5,   20. ,   21.5])
    expected = np.array( np.hstack( (expected_x[:,np.newaxis],
                                   expected_y[:,np.newaxis])))
    N_close = 0
    N_total_detected = len(actual_x)
    N_total_possible = len(expected_x)

    dist_threshold = 5 # should be within 5 pixels
    fraction_same_threshold = 0.9

    fraction_different_threshold = 1.0 - fraction_same_threshold
    for i in range(len(actual)):
        this_pt = actual[i]
        dists = np.sum((this_pt - expected)**2,axis=1)
        closest_dist = np.min(dists)

        if closest_dist < dist_threshold:
            N_close +=1

    frac=N_close/float(N_total_possible)
    assert abs(frac-1.0) < fraction_different_threshold

def prune_non_simply_connected(similar_direction_graphs):
    filtered = []
    for graph in similar_direction_graphs:
        periphery = get_singly_connected_nodes( graph )
        if len(periphery)>2:
            print ('WARNING: graph has > 2 nodes in periphery; '
                   'image analysis suspect; discarding bad graph. Hint: '
                   'try decreasing "angle_precision_degrees" in .cfg file.')
            #print ' periphery',periphery
            #print ' graph',graph.edges()
            if 0 and debug_line_finding:
                pylab.figure()
                bad_graph_edges = graph.edges()
                import networkx
                import networkx.drawing.nx_pylab as nx_pylab
                g = networkx.Graph()
                for e in bad_graph_edges:
                    g.add_edge(*e)
                networkx.drawing.nx_pylab.draw(g)
        else:
            filtered.append( graph )
    return filtered

def get_similar_direction_graphs(fmf,frame,
                                 use='raw',return_early=False,
                                 debug_line_finding=False,
                                 aspect_ratio = 1.0,
                                 direction_eps_radians=None,
                                 chess_preview=False,
                                 ):
    bg_im,tmp = fmf.get_frame(0)
    bg_im = imops.to_mono8(fmf.get_format(),bg_im)
    imnx_orig,tmp = fmf.get_frame(frame)
    imnx_orig = imops.to_mono8(fmf.get_format(),imnx_orig)

    imnx_no_bg = get_non_background( imnx_orig, bg_im )
    imnx_binary = binarize(imnx_no_bg)
    imnx_rawbinary = binarize(imnx_orig)
    if use == 'no_bg':
        imnx_use = imnx_no_bg
    elif use == 'binary':
        imnx_use = imnx_binary
    elif use == 'rawbinary':
        imnx_use = imnx_rawbinary
    elif use == 'raw':
        imnx_use = imnx_orig
    else:
        raise ValueError('unknown use image')

    if chess_preview:
        pylab.imshow(imnx_use)
        pylab.title('preview of chessboard finding image - close to continue')
        pylab.show()

    x,y=extract_corners(imnx_use)
    if len(x) == 0:
        raise ValueError('extract corners found no corners. cannot continue')

    if chess_preview:
        pylab.imshow(imnx_use)
        pylab.plot(x,y,'wo')
        pylab.title('found corners')
        pylab.show()

    show_clusters_frame = frame
    graph, nodes = points2graph(x, y,
                                show_clusters=debug_line_finding,
                                show_clusters_frame=show_clusters_frame,
                                aspect_ratio = aspect_ratio,
                                )
    if return_early:
        print 'returning early with entire super-graph'
        similar_direction_graphs = [graph]
        return (similar_direction_graphs, imnx_orig, imnx_no_bg, imnx_binary,
                imnx_use, imnx_rawbinary)

    similar_direction_graphs = [] # collection of all the graphs of similar directions
    for node in nodes:

        subgraph = find_subgraph_similar_direction(
            graph,
            source=node,
            direction_eps_radians=direction_eps_radians,
            already_done=similar_direction_graphs)
        if subgraph is not None:
            similar_direction_graphs.append( subgraph )

        if 0:
            print 'backwards search?!'
            # do again to get other direction
            subgraph = find_subgraph_similar_direction(
                graph,
                source=node,
                direction_eps_radians=direction_eps_radians,
                already_done=similar_direction_graphs)
            if subgraph is not None:
                similar_direction_graphs.append( subgraph )

    # filter to force 2 or more edges in graph
    similar_direction_graphs = [ graph for graph in similar_direction_graphs if
                                 len(graph.edges()) >= 2 ]

    if debug_line_finding:
        print 'edges in each subgraph (frame %d)'%frame,'-'*40
        for graph in similar_direction_graphs:
            print graph.edges()
        print '-'*40
        print

    return (similar_direction_graphs, imnx_orig, imnx_no_bg, imnx_binary,
            imnx_use, imnx_rawbinary)

def main():
    parser = OptionParser(usage='%prog CONFIG_FILE',
                          version="%prog 0.1")

    parser.add_option("--view-results", action='store_true',
                      default=False)

    parser.add_option("--view-results-quick", action='store_true',
                      default=False)

    parser.add_option("--show-chessboard-finder-preview",
                      help=("show the image being fed to chessboard corner "
                            "finder"),
                      action='store_true',
                      default=False)

    parser.add_option("--find-and-show1",
                      help=("find checkerboard intersections and "
                            "display them (don't compute distortion)"),
                      action='store_true',
                      default=False)

    parser.add_option("--find-and-show2",
                      help=("find checkerboard intersections and "
                            "display them (don't compute distortion)"),
                      action='store_true',
                      default=False)

    parser.add_option("--debug-line-finding",
                      help=("show the line finding clustering data"),
                      action='store_true',
                      default=False)

    parser.add_option("--debug-nodes",
                      help="print to console the node numbers and edges",
                      action='store_true',
                      default=False)

    (cli_options, args) = parser.parse_args()
    if not len(args)==1:
        raise RuntimeError('one command-line argument is needed - '
                           'the configFile')
    configFile = args[0]

    defaults = dict(
        # keep flydra-sphinx-docs/calibration.rst up to date
        use = 'raw',
        angle_precision_degrees=10.0,
        aspect_ratio = 1.0,

        show_lines = False,
        return_early=False,
        debug_line_finding = False,
        epsfcn = 1e-9,
        print_debug_info = False,
        save_debug_images = False,

        ftol=0.001,
        xtol=0,
        do_plot = False,

	K13 = None, # center guess X
	K23 = None, # center guess Y

	kc1 = 0.0, # initial guess of radial distortion
	kc2 = 0.0, # initial guess of radial distortion
        )

    if cli_options.find_and_show1:
        defaults['return_early'] = True
        defaults['do_plot'] = True

    if cli_options.find_and_show2:
        defaults['do_plot'] = True

    cli_options.find_and_show = (cli_options.find_and_show1 or
                                 cli_options.find_and_show2)

    if cli_options.debug_line_finding:
        defaults['debug_line_finding'] = True
        defaults['do_plot'] = True

    configFile = os.path.abspath( configFile)
    if not os.path.exists( configFile ):
        raise RuntimeError('could not read config file %s'%configFile)

    config_file_results = {}
    execfile(configFile,globals(),config_file_results)

    config = defaults.copy()
    config.update( config_file_results )


    class OptionsClass:
        def __init__(self, mydict):
            for key,value in mydict.iteritems():
                setattr( self, key, value )
    options = OptionsClass(config)

    if cli_options.view_results:
        options.do_plot = True

    if cli_options.view_results_quick:
        helper = reconstruct_utils.make_ReconstructHelper_from_rad_file(
            options.rad_fname)
        pylab.figure()
        ax = pylab.subplot(1,1,1)
        visualize_distortions( ax, helper)
        pylab.title( options.rad_fname )
        pylab.show()
        sys.exit(0)

    all_graphs = []
    graph_idxs_by_frames = []
    all_imnx_use = []

    fmf = FlyMovieFormat.FlyMovie(options.fname)

    for frame in options.frames:
        (similar_direction_graphs, imnx_orig, imnx_no_bg, imnx_binary,
         imnx_use, imnx_rawbinary) = get_similar_direction_graphs(
            fmf,frame,
            use=options.use,
            return_early=options.return_early,
            debug_line_finding = options.debug_line_finding,
            aspect_ratio = options.aspect_ratio,
            direction_eps_radians=options.angle_precision_degrees*D2R,
            chess_preview=cli_options.show_chessboard_finder_preview,
            )

        if 1:
            filtered = prune_non_simply_connected(similar_direction_graphs)
            print '%d of %d original graphs survived'%(
                len(filtered),len(similar_direction_graphs))
            similar_direction_graphs = filtered
            print 'mean N nodes: %f'%np.mean([len(g.nodes()) for g in similar_direction_graphs])

        start_idx = len(all_graphs)
        all_graphs.extend( similar_direction_graphs )
        stop_idx = len(all_graphs)
        graph_idxs_by_frames.append( range(start_idx,stop_idx) )
        all_imnx_use.append( imnx_use )

        if cli_options.debug_nodes:
            print 'edges for frame %d ================'%frame
            for subgraph in similar_direction_graphs:
                print subgraph.edges()
            print

        # XXX uses last image
    similar_direction_graphs = all_graphs
    if len(all_graphs)==0:
        raise ValueError(
            'no valid graphs were found. Cannot continue')

    did_plot = False
    if options.do_plot:

        frame_mpl_figures = {}
        # plot original, distorted image
        for j, frame_no in enumerate(options.frames):
            color_count = 0
            plot_nodes = []
            frame_mpl_figures[frame_no] = pylab.figure()
            if (options.return_early or cli_options.find_and_show):
                im_ax = pylab.subplot(1,1,1)
            else:
                im_ax = frame_mpl_figures[frame_no].add_subplot(2,1,1)
            for i in graph_idxs_by_frames[j]: # get points for last image
                subgraph = similar_direction_graphs[i]
                if options.debug_line_finding or cli_options.debug_nodes:
                    pos = {}
                    for node in subgraph.nodes():
                        pos[node] = node.get_pos()

                    NX.draw(subgraph,
                            pos=pos,
                            width=2,
                            )
                else:
                    if 0:
                        xys = numpy.array([ node.get_pos() for node in subgraph.nodes() ])
                        im_ax.plot( xys[:,0], xys[:,1], 'bo' )
                    else:
                        color = get_color(color_count)
                        color_count += 1
                        for edge in subgraph.edges():
                            ## xys = numpy.array(
                            ##     [ edge[i].get_rand_pos(repr(subgraph)) for i in [0,1] ])
                            xys = numpy.array([ edge[i].get_pos() for i in [0,1] ])
                            im_ax.plot( xys[:,0], xys[:,1], '%so-'%color, mew=0)
                            plot_nodes.append( edge[0] )
                            plot_nodes.append( edge[1] )
            if 1:
                plot_nodes = list(set(plot_nodes)) # unique
                for node in plot_nodes:
                    x,y=node.get_pos()
                    im_ax.text(x,y,'%s'%node)

                im_ax.set_title('frame %d - original (distorted)'%options.frames[j])

                if options.show_lines:
                    # draw line fits
                    xys = numpy.array([ node.get_pos() for node in subgraph.nodes() ])
                    x, residues, rank, s = fit_line( xys )
                    xi = numpy.linspace(0,700,200)
                    yi = x[0]*xi + x[1]
                    im_ax.plot( xi, yi, 'w-', lw=2 )


            if 1:
                im_ax.imshow( all_imnx_use[j],
                              origin='lower',
                              interpolation='nearest',
                              cmap=pylab.cm.pink,
                              )

        if not (options.return_early or cli_options.find_and_show):
            grid_figure = pylab.figure()
            distorted_grid_axes = grid_figure.add_subplot(3,1,1)
            for j,frame_idxs in enumerate(graph_idxs_by_frames):
                color = get_color(j)
                did_frame_label = False
                for i in frame_idxs:
                    subgraph = similar_direction_graphs[i]
                    if 0:
                        # bad - plots points as lines
                        xys = numpy.array([ node.get_pos() for node in subgraph.nodes() ])
                        if not did_frame_label:
                            kwargs = dict(label='%d'%(options.frames[j],))
                            did_frame_label = True
                        else:
                            kwargs = dict()
                        print 'subgraph.edges()',subgraph.edges()
                        distorted_grid_axes.plot( xys[:,0], xys[:,1], '%s-'%color, **kwargs )
                    else:
                        for edge in subgraph.edges():
                            xys = numpy.array([ edge[i].get_pos() for i in [0,1] ])
                            if not did_frame_label:
                                kwargs = dict(label='%d'%(options.frames[j],))
                                did_frame_label = True
                            else:
                                kwargs = dict()
                            #print 'subgraph.edges()',subgraph.edges()
                            distorted_grid_axes.plot( xys[:,0], xys[:,1], '%s-'%color, **kwargs )

            distorted_grid_axes.set_title('grid original (distorted)')
            if len(options.frames)>1:
                distorted_grid_axes.legend()

        did_plot = True

    if options.return_early or cli_options.find_and_show:
        pylab.show()
        sys.exit()

    height, width = imnx_use.shape
    if not cli_options.view_results:
        obj = Objective( similar_direction_graphs,#[:2],
                         width=width,
                         height=height,
                         debug = options.print_debug_info,
                         save_debug_images = options.save_debug_images,
                         aspect_ratio=options.aspect_ratio,
                         )

        p0 = obj.get_default_p0(config)
        print 'p0',p0[:4]

        if 0:
            pylab.figure()
            ax = pylab.gca()
            obj.plot_fit(ax, p0)
            pylab.show()
            sys.exit()

        initial_err = obj.sumsq_err(p0)
        if numpy.isnan(initial_err):
            raise ValueError('initial error evaluation failed')
        print 'initial_err',initial_err
        print 'calculating results for file %s ...'%(options.rad_fname)

        if 1:
            # call optimizer
            results = scipy.optimize.minpack.leastsq(
                obj.lm_err_func,
                numpy.array(p0,copy=True), # workaround bug (scipy ticket 637)
                epsfcn=options.epsfcn,
                ftol=options.ftol,
                xtol=options.xtol,
                maxfev=int(1e6),
                full_output=True,
                )
            pfinal, cov_x, infodict, mesg, ier = results
            print
            print '%d function calls'%(infodict['nfev'],)
            print 'covariance of parameters:'
            print cov_x
            if ier in (1,2,3,4):
                print 'Solution found.',mesg
            else:
                print 'WARNING: Solution not found!',mesg

        helper = obj.get_helper_for_params( pfinal )
        mesg = mesg.replace('\n',' ')

        print 'pfinal',repr(pfinal[:4])
        final_err=obj.sumsq_err( pfinal )
        helper.save_to_rad_file( options.rad_fname, comments = 'final err %.1f, leastsq result %d: %s'%(final_err, ier, mesg) )
        print 'final_err',final_err
        if numpy.allclose(final_err,initial_err):
            print 'WARNING: no improvement after fitting. Reduce tolerance ("ftol","xtol") and try again.'

    else:
        # cli_options.view_results == True
        helper = reconstruct_utils.make_ReconstructHelper_from_rad_file(options.rad_fname)

    if options.do_plot:
        # plot corrected, un-distorted images
        for j, frame_no in enumerate(options.frames):
            im_ax = frame_mpl_figures[frame_no].add_subplot(2,1,2)
            for i in graph_idxs_by_frames[j]: # get points for last image
                subgraph = similar_direction_graphs[i]

                if 0:
                    orig_xys = numpy.array([ node.get_pos() for node in subgraph.nodes() ])
                    xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
                    im_ax.plot( xys[:,0], xys[:,1], 'bo' )
                else:
                    for edge in subgraph.edges():
                        orig_xys = numpy.array([ edge[i].get_pos() for i in [0,1] ])
                        xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
                        im_ax.plot( xys[:,0], xys[:,1], 'bo-')

                if options.show_lines:
                    orig_xys = numpy.array([ node.get_pos() for node in subgraph.nodes() ])
                    xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
                    x, residues, rank, s = fit_line( xys )
                    xi = numpy.linspace(0,700,200)
                    yi = x[0]*xi + x[1]
                    im_ax.plot( xi, yi, 'b-', lw=2 )

            if 1:
                imnx2 = helper.undistort_image(all_imnx_use[j] )

                im_ax.imshow( imnx2,
                              origin='lower',
                              interpolation='nearest',
                              cmap=pylab.cm.pink,
                              )
            im_ax.set_title('frame %d - corrected (undistorted)'%options.frames[j])

        undistorted_grid_axes = grid_figure.add_subplot(3,1,2,
                                                        sharex=distorted_grid_axes,
                                                        sharey=distorted_grid_axes,
                                                        )
        for j,frame_idxs in enumerate(graph_idxs_by_frames):
            color = get_color(j)
            did_frame_label = False



            for i in frame_idxs:
                subgraph = similar_direction_graphs[i]

                for edge in subgraph.edges():
                    orig_xys = numpy.array([ edge[i].get_pos() for i in [0,1] ])
                    xys = numpy.array([helper.undistort( ox, oy ) for (ox,oy) in orig_xys])
                    if not did_frame_label:
                        kwargs = dict(label='%d'%(options.frames[j],))
                        did_frame_label = True
                    else:
                        kwargs = dict()
                    undistorted_grid_axes.plot( xys[:,0], xys[:,1], '%s-'%color, **kwargs )

        K = helper.get_K()
        x0 = K[0,2]
        y0 = K[1,2]
        undistorted_grid_axes.plot( [x0], [y0], 'ko' )
        print 'x0,y0',x0,y0

        undistorted_grid_axes.set_title('grid corrected (undistorted)')

        if len(options.frames)>1:
            undistorted_grid_axes.legend()

        if 1:
            ax = grid_figure.add_subplot(3,1,3)
            visualize_distortions( ax, helper,
                                   width=width,
                                   height=height)
        did_plot = True

    if did_plot:
        pylab.show()


if __name__ == '__main__':
    main()
