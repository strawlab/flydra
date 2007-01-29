from __future__ import division
import numpy
import tables as PT
import scipy.io
import sys

def main():
    filename = sys.argv[1]
    do_it(filename=filename)

def do_it(filename=None):
    kresults = PT.openFile(filename,mode="r")
    print 'reading files...'
    if 0:
        table1 = kresults.root.kalman_estimates.read(start=0,stop=5,flavor='numpy')
        table2 = kresults.root.kalman_observations.read(start=0,stop=5,flavor='numpy')
    else:
        table1 = kresults.root.kalman_estimates.read(flavor='numpy')
        table2 = kresults.root.kalman_observations.read(flavor='numpy')
    print 'done.'
    kresults.close()
    del kresults

    if 1:
        obj_ids = table1.field('obj_id').copy()
        obj_ids = numpy.unique(obj_ids)

        obs_cond = None
        k_cond = None

        for obj_id in obj_ids:
            this_obs_cond = table2.field('obj_id') == obj_id
            n_observations = numpy.sum(this_obs_cond)
            if n_observations > 10:
                if obs_cond is None:
                    obs_cond = this_obs_cond
                else:
                    obs_cond = obs_cond | this_obs_cond

                this_k_cond = table1.field('obj_id') == obj_id
                if k_cond is None:
                    k_cond = this_k_cond
                else:
                    k_cond = k_cond | this_k_cond
                    
        table1 = table1[k_cond]
        table2 = table2[obs_cond]

        newfilename = filename + '-short-only.mat'
    else:
        newfilename = filename + '.mat'
    
    data = dict( kalman_obj_id = table1.field('obj_id'),
                 kalman_frame = table1.field('frame'),
                 kalman_x = table1.field('x'),
                 kalman_y = table1.field('y'),
                 kalman_z = table1.field('z'),
                 kalman_xvel = table1.field('xvel'),
                 kalman_yvel = table1.field('yvel'),
                 kalman_zvel = table1.field('zvel'),
                 kalman_xaccel = table1.field('xaccel'),
                 kalman_yaccel = table1.field('yaccel'),
                 kalman_zaccel = table1.field('zaccel'),
                 observation_obj_id = table2.field('obj_id'),
                 observation_frame = table2.field('frame'),
                 observation_x = table2.field('x'),
                 observation_y = table2.field('y'),
                 observation_z = table2.field('z') )

    if 1:
        print "converting int32 to float64 to avoid scipy.io.mio.savemat bug"
        for key in data:
            #print 'converting field',key, data[key].dtype, data[key].dtype.char
            if data[key].dtype.char == 'l':
                data[key] = data[key].astype(numpy.float64)

    scipy.io.mio.savemat(newfilename,data)
