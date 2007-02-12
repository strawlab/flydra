from __future__ import division
import detect_saccades
import stimulus_positions
from conditions import files, condition_names, stim_names
import tables
import numpy
import pickle

if 1:
    by_filename = {}
    ca = detect_saccades.CachingAnalyzer()
    #for condition_name in condition_names:
    for condition_name in ['double post']:
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
            for obj_id_enum,obj_id in enumerate(use_obj_ids):
                if obj_id_enum%100 == 0:
                    print 'object %d of %d'%(obj_id_enum,len(use_obj_ids))
                    
##                if obj_id != 158:
##                    continue
                
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
                for lineseg in all_stim_verts:
                    xavg = (lineseg[0][0] + lineseg[1][0])/2
                    yavg = (lineseg[0][1] + lineseg[1][1])/2

##                    print 'xavg, yavg',xavg, yavg
##                    print 'verts[:10]',verts[:10]
                    xdist = verts[:,0]-xavg
                    ydist = verts[:,1]-yavg
##                    print 'xdist[:10]',xdist[:10]
##                    print 'ydist[:10]',ydist[:10]

                    dist = numpy.sqrt(xdist**2+ydist**2)
                    mindist = min(mindist,float(dist.min()))
                    #print 'mindist',mindist
                if mindist<0.030 and len(verts) >1000:
                    print obj_id, len(verts), mindist
                    
                    # find 
                min_dist_by_obj_id[obj_id] = mindist
                min_dists.append(mindist)
##                if obj_id_enum > 150:
##                    break
            min_dists = numpy.array(min_dists)
            min_dists_order = numpy.argsort(min_dists)
            min_dists_order = min_dists_order[:20]
            use_obj_ids = numpy.array(use_obj_ids)
            closest_flights_obj_ids = use_obj_ids[min_dists_order]
            print closest_flights_obj_ids[:20]

            by_filename[filename] = closest_flights_obj_ids

            fd = open('closest.pkl',mode='wb')
            pickle.dump(by_filename,fd)
            fd.close()
