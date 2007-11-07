import numpy
import os, sys, time#, pprint
from flydra_tracker import Tracker
import cProfile
import lsprofcalltree

def kalmanize(src_filename,
              max_iterations=None,
              ):

    import pickle
    
    fd = open(src_filename,mode="rb")
    loaded = pickle.load(fd)
    fd.close()
    
    frames = []
    start = time.time()
    for i,tup in enumerate(loaded):
        if tup[0] == 'gob':
            corrected_framenumber,s_pluecker_coords_by_camn,camn2cam_id = tup[1]
            frames.append( corrected_framenumber )
            pluecker_coords_by_camn = pickle.loads(s_pluecker_coords_by_camn)
            tracker.gobble_2d_data_and_calculate_a_posteri_estimates(corrected_framenumber,pluecker_coords_by_camn,camn2cam_id)
        elif tup[0] == 'ntrack':
            assert len(tracker.live_tracked_objects)==tup[1]
        elif tup[0] == 'join':
            (corrected_framenumber,
             this_observation_orig_units,
             this_observation_camns,
             this_observation_idxs
             ) = tup[1]
            tracker.join_new_obj( corrected_framenumber,
                                  this_observation_orig_units,
                                  this_observation_camns,
                                  this_observation_idxs
                                  )
        elif tup[0] == 'tracker':
            tracker = tup[1]
        else:
            raise ValueError('unknown code %s'%tup[0])
        
        if max_iterations is not None:
            if i>=max_iterations:
                break
            
    stop = time.time()
    print len(frames),'frames'
    print 'fps:',len(frames)/(stop-start)
    
def main():
    src_filename = sys.argv[1]
    kalmanize(src_filename)

if __name__=='__main__':
    #main()
    p = cProfile.Profile()
    p.run('main()')
    k = lsprofcalltree.KCacheGrind(p)
    data = open('prof.kgrind', 'w+')
    k.output(data)
    data.close()
