import numpy
import pickle
import pylab
from conditions import files, condition_names
import tables as PT

min_length = 15*100 # 15 second trajectories

if 1:
    if 1:
        fname = 'trajectory_lengths.pkl'
        fd = open(fname,mode='rb')
        results = pickle.load(fd)
        fd.close()

        all_mean_speeds = []
        for condition_name in condition_names:
            print condition_name
            filenames = files[condition_name]
            for filename in filenames:
                print filename,
                this_results = results[condition_name][filename]
                trajectory_lengths = this_results.keys()
                trajectory_lengths.sort()
                trajectory_lengths.reverse()
                
                kresults = PT.openFile(filename,mode="r")
                obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
                this_mean_speeds = []
                max_mean = 0
                for trajectory_length in trajectory_lengths:
                    if trajectory_length < min_length:
                        break
                    
                    use_obj_ids = this_results[trajectory_length]
                    
                    for obj_id in use_obj_ids:
                        idxs = numpy.nonzero(obj_ids == obj_id)[0]
                        
                        xvels = kresults.root.kalman_estimates.readCoordinates(idxs,field='xvel',flavor='numpy')
                        yvels = kresults.root.kalman_estimates.readCoordinates(idxs,field='yvel',flavor='numpy')
                        zvels = kresults.root.kalman_estimates.readCoordinates(idxs,field='zvel',flavor='numpy')
                        
                        vels = numpy.vstack((xvels,yvels,zvels)).T
                        speeds = numpy.sqrt(numpy.sum(vels**2,axis=1))

                        mean_speed = numpy.mean(speeds)
                        this_mean_speeds.append(mean_speed)

                        max_mean = max(max_mean,mean_speed)
                        if max_mean == mean_speed:
                            max_mean_obj = obj_id
                        

                print numpy.mean(this_mean_speeds), len(this_mean_speeds), min(this_mean_speeds), max(this_mean_speeds), max_mean_obj
                all_mean_speeds.extend(this_mean_speeds)
                kresults.close()
        all_mean_speeds = numpy.array(all_mean_speeds)
        print all_mean_speeds
