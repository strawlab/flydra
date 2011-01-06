from __future__ import division
import numpy
import tables as PT
import scipy.io
import sys
import tables.flavor
tables.flavor.restrict_flavors(keep=['numpy'])

def main():
    filename = sys.argv[1]
    do_it(filename=filename)

def do_it(filename=None,
          rows=None,
          ignore_observations=False,
          min_num_observations=10,
          newfilename=None,
          extra_vars=None):

    if filename is None and rows is None:
        raise ValueError("either filename or rows must be set")

    if filename is not None and rows is not None:
        raise ValueError("either filename or rows must be set, but not both")

    if extra_vars is None:
        extra_vars = {}

    if filename is not None:
        kresults = PT.openFile(filename,mode="r")
        print 'reading files...'
        table1 = kresults.root.kalman_estimates.read()
        if not ignore_observations:
            table2 = kresults.root.kalman_observations.read()
        print 'done.'
        kresults.close()
        del kresults

    if rows is not None:
        table1 = rows
        if not ignore_observations:
            raise ValueError("no observations can be saved in rows mode")

    if not ignore_observations:
        obj_ids = table1['obj_id']
        obj_ids = numpy.unique(obj_ids)

        obs_cond = None
        k_cond = None

        for obj_id in obj_ids:
            this_obs_cond = table2['obj_id'] == obj_id
            n_observations = numpy.sum(this_obs_cond)
            if n_observations > min_num_observations:
                if obs_cond is None:
                    obs_cond = this_obs_cond
                else:
                    obs_cond = obs_cond | this_obs_cond

                this_k_cond = table1['obj_id'] == obj_id
                if k_cond is None:
                    k_cond = this_k_cond
                else:
                    k_cond = k_cond | this_k_cond

        table1 = table1[k_cond]
        table2 = table2[obs_cond]

        if newfilename is None:
            newfilename = filename + '-short-only.mat'
    else:
        if newfilename is None:
            newfilename = filename + '.mat'

    data = dict( kalman_obj_id = table1['obj_id'],
                 kalman_frame = table1['frame'],
                 kalman_x = table1['x'],
                 kalman_y = table1['y'],
                 kalman_z = table1['z'],
                 kalman_xvel = table1['xvel'],
                 kalman_yvel = table1['yvel'],
                 kalman_zvel = table1['zvel'])

    if 'dir_x' in table1.dtype.fields:
        for d in ('dir_x','dir_y','dir_z'):
            data[d] = table1[d]

    if 'xaccel' in table1:
        # acceleration state not in newer dynamic models
        dict2 = dict(
                 kalman_xaccel = table1['xaccel'],
                 kalman_yaccel = table1['yaccel'],
                 kalman_zaccel = table1['zaccel'])
        data.update(dict2)

    if not ignore_observations:
        extra = dict(
            observation_obj_id = table2['obj_id'],
            observation_frame = table2['frame'],
            observation_x = table2['x'],
            observation_y = table2['y'],
            observation_z = table2['z'] )
        data.update(extra)

    if 0:
        print "converting int32 to float64 to avoid scipy.io.savemat bug"
        for key in data:
            #print 'converting field',key, data[key].dtype, data[key].dtype.char
            if data[key].dtype.char == 'l':
                data[key] = data[key].astype(numpy.float64)

    for key,value in extra_vars.iteritems():
        if key in data:
            print 'WARNING: requested to save extra variable %s, but already in data, not overwriting'%key
            continue
        data[key] = value

    scipy.io.savemat(newfilename,data,appendmat=False)

if __name__=='__main__':
    print "WARNING: are you sure you want to run this program and not 'data2smoothed'?"
    main()
