from __future__ import division
import detect_saccades
import stimulus_positions
from conditions import files, condition_names, stim_names
import tables
import numpy
import pickle

if 1:
    must_have_been_dist_away = 0.05 # 50 mm
    print 'must_have_been_dist_away',must_have_been_dist_away
    
    #mode = 'all'
    #mode = 'takeoff' # not really takeoff or landing, but distance of first and last tracked points, respectively
    mode = 'landing'
    
    by_filename = {}
    ca = detect_saccades.CachingAnalyzer()
    #for condition_name in condition_names:
    for condition_name in ['d2']:
        filenames = files[condition_name]
        stim = stim_names[condition_name]
        all_stim_verts = stimulus_positions.stim_positions[stim]

        for filename in filenames:
            print 'filename',filename,condition_name
            kresults = tables.openFile(filename,mode='r')
            obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
            use_obj_ids = numpy.unique(obj_ids)

            min_dist_by_obj_id = {}
            min_dists = []
            use_idxs = []
            for obj_id_enum,obj_id in enumerate(use_obj_ids):
##                if obj_id>1000:
##                    break
                
                if obj_id_enum%100 == 0:
                    print 'object %d of %d'%(obj_id_enum,len(use_obj_ids))
                    
                verts = ca.get_raw_positions(obj_id,
                                             kresults)
                if len(verts)<20:
                    continue

                z = verts[:,2]
                narrow_z_cond = (z >= 0.05) & (z <= 0.25)
                verts = verts[narrow_z_cond]

                if len(verts) == 0:
                    continue
                
                mindist = numpy.inf
                maxdist = 0
                for lineseg in all_stim_verts:
                    
                    xavg = (lineseg[0][0] + lineseg[1][0])/2
                    yavg = (lineseg[0][1] + lineseg[1][1])/2

                    xdist = verts[:,0]-xavg
                    ydist = verts[:,1]-yavg

                    dist = numpy.sqrt(xdist**2+ydist**2)
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
    kresults.close()
    
    fd = open('closest.pkl',mode='wb')
    pickle.dump(by_filename,fd)
    fd.close()
            
