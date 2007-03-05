from __future__ import division
import detect_saccades
import stimulus_positions
from conditions import files, condition_names, stim_names
import tables
import numpy
import scipy.io
import pickle, os


def calc_dist_old(lineseg,verts):
    xavg = (lineseg[0][0] + lineseg[1][0])/2
    yavg = (lineseg[0][1] + lineseg[1][1])/2

    xdist = verts[:,0]-xavg
    ydist = verts[:,1]-yavg

    dist = numpy.sqrt(xdist**2+ydist**2)
    return dist

def calc_dist(lineseg,verts):
    """return minimum distance between lineseg and vertices

In threespace (ascii art):

 u and v vectors are vertices of lineseg.
 w vector(s) are verts.

then:

 a = v-u
 b = w-u

we can find:

 x = (projection of b onto a) + u
 
 x is the projection
 
   * v
   |
   |
   |
   |
 x *   * w
   |  /
   | /
   |/
   * u


    """
    def vec_abs(x):
        return numpy.sqrt(numpy.sum( x**2, axis=0 ))
    v = numpy.asarray(lineseg[0])
    u = numpy.asarray(lineseg[1])

    w = numpy.asarray(verts).T
    n_pts = w.shape[1]

    a = v-u
    b = w-u[:,numpy.newaxis]

    amag = vec_abs(a)
    anorm = a/amag
    
    len_b_projected_on_a = numpy.dot(anorm,b)

    # closest point on axis of lineseg:
    x = len_b_projected_on_a[numpy.newaxis,:]*anorm[:,numpy.newaxis] + u[:,numpy.newaxis]
    perpendicular_dist = vec_abs(w-x)

    u_dist = vec_abs(w-u[:,numpy.newaxis])
    v_dist = vec_abs(w-v[:,numpy.newaxis])
    endpoint_dist = numpy.where( u_dist < v_dist, u_dist, v_dist )

    midpoint_closest_cond = (len_b_projected_on_a <= amag) & (len_b_projected_on_a >= 0)
    dist = numpy.where( midpoint_closest_cond, perpendicular_dist, endpoint_dist )

    if 0:
        print 'u',u
        print 'v',v
        print
        print 'w[:,:4]'
        print w[:,:4]
        print
        print 'len_b_projected_on_a[:4]'
        print len_b_projected_on_a[:4]
        print
        print 'a'
        print a
        print
        print 'len_b_projected_on_a[numpy.newaxis,:]*anorm[:,numpy.newaxis]'
        print (len_b_projected_on_a[numpy.newaxis,:]*anorm[:,numpy.newaxis])[:,:4]
        print
        print 'x[:,:4]'
        print x[:,:4]
        print
        print 'u_dist[:4]'
        print u_dist[:4]
        print
        print 'v_dist[:4]'
        print v_dist[:4]
        print
        print 'perpendicular_dist[:4]'
        print perpendicular_dist[:4]
        print
        print 'dist[:4]'
        print dist[:4]
        print
        1/0
    return dist

if 1:
    must_have_been_dist_away = 0.05 # 50 mm
    print 'must_have_been_dist_away',must_have_been_dist_away
    
    #mode = 'all'
    #mode = 'takeoff' # not really takeoff or landing, but distance of first and last tracked points, respectively
    mode = 'landing'
    
    by_filename = {}
    ca = detect_saccades.CachingAnalyzer()
    #for condition_name in condition_names:
    for condition_name in ['half no odor, w/ wind']:
        filenames = files[condition_name]
        stim = stim_names[condition_name]
        all_stim_verts = stimulus_positions.stim_positions[stim]

        for filename in filenames:
            print 'filename',filename,condition_name

            if os.path.splitext(filename)[1] == '.mat':
                mat_data = scipy.io.mio.loadmat(filename)
                obj_ids = mat_data['kalman_obj_id']
                obj_ids = obj_ids.astype( numpy.uint32 )
                obs_obj_ids = obj_ids # use as observation length, even though these aren't observations
                use_obj_ids = numpy.unique(obj_ids)
                is_mat_file = True
                data_file = mat_data
            else:
                kresults = tables.openFile(filename,mode='r')
                obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
                use_obj_ids = numpy.unique(obj_ids)
                is_mat_file = False
                data_file = kresults

            min_dist_by_obj_id = {}
            min_dists = []
            use_idxs = []
            for obj_id_enum,obj_id in enumerate(use_obj_ids):
                if obj_id_enum%100 == 0:
                    print 'object %d of %d'%(obj_id_enum,len(use_obj_ids))
                verts = ca.get_raw_positions(obj_id,
                                             data_file)
                if len(verts)<20:
                    continue

                if 0:
                    z = verts[:,2]
                    narrow_z_cond = (z >= 0.05) & (z <= 0.25)
                    verts = verts[narrow_z_cond]

                if len(verts) == 0:
                    continue
                
                mindist = numpy.inf
                maxdist = 0
                for lineseg in all_stim_verts:
                    #dist = calc_dist_old(lineseg,verts)
                    dist = calc_dist(lineseg,verts)
                    if dist.max() < must_have_been_dist_away:
                        continue

                    if (dist[0] < must_have_been_dist_away):
                        # probably takeoff
                        if mode in ['all','takeoff']:
                            mindist = min(mindist,float(dist.min()))
                    elif (dist[-1] < must_have_been_dist_away):
                        if mode in ['all','landing']:
                            mindist = min(mindist,float(dist.min()))
                    elif mode in ['all']:
                        mindist = min(mindist,float(dist.min()))
                    
                if mindist<0.030 and len(verts) >1000:
                    print obj_id, len(verts), mindist
                min_dist_by_obj_id[obj_id] = mindist
                min_dists.append(mindist)
                use_idxs.append(obj_id_enum)
            min_dists = numpy.array(min_dists)
            use_idxs = numpy.array(use_idxs)
            use_obj_ids = numpy.array(use_obj_ids)
            
            min_dists_order = numpy.argsort(min_dists)
            min_dists_order = min_dists_order[:50]
            closest_flights_idxs = use_idxs[min_dists_order]
            closest_flights_obj_ids = use_obj_ids[closest_flights_idxs]

            by_filename[filename] = closest_flights_obj_ids

    ca.close()
    if not is_mat_file:
        kresults.close()
    
    fd = open('closest.pkl',mode='wb')
    pickle.dump(by_filename,fd)
    fd.close()
            
