from __future__ import division
import tables
import numpy
import math
import scipy.io
DEBUG = False

import adskalman
import flydra.kalman.dynamic_models
import flydra.kalman.params
import flydra.kalman.flydra_kalman_utils

# global
printed_dynamics_name = False

def find_peaks(y,threshold,search_cond=None):
    """find local maxima above threshold in data y

    returns indices of maxima
    """
    if search_cond is None:
        search_cond = numpy.ones(y.shape,dtype=numpy.bool)

    nz = numpy.nonzero(search_cond)[0]
    ysearch = y[search_cond]
    if len(ysearch)==0:
        return []
    peak_idx_search = numpy.argmax(ysearch)
    peak_idx = nz[peak_idx_search]
    
    # descend y in positive x direction
    new_idx = peak_idx
    curval = y[new_idx]
    if curval < threshold:
        return []
    
    while 1:
        new_idx += 1
        if new_idx>=len(y):
            break
        newval = y[new_idx]
        if newval>curval:
            break
        curval= newval
    max_idx = new_idx-1

    # descend y in negative x direction
    new_idx = peak_idx
    curval = y[new_idx]
    while 1:
        new_idx -= 1
        if new_idx < 0:
            break
        newval = y[new_idx]
        if newval>curval:
            break
        curval= newval
    min_idx = new_idx+1

    this_peak_idxs = numpy.arange(min_idx,max_idx+1)
    new_search_cond = numpy.array(search_cond,copy=True)
    new_search_cond[this_peak_idxs] = 0

    all_peak_idxs = [peak_idx]
    all_peak_idxs.extend( find_peaks(y,threshold,search_cond=new_search_cond) )
    return all_peak_idxs

def my_decimate(x,q):
    if q==1:
        return x
    
    if 0:
        from scipy_utils import decimate as matlab_decimate # part of ads_utils, contains code translated from MATLAB
        return matlab_decimate(x,q)
    elif 0:
        # take qth point
        return x[::q]
    else:
        # simple averaging
        xtrimlen = int(math.ceil(len(x)/q))*q
        lendiff = xtrimlen-len(x)
        xtrim = numpy.zeros( (xtrimlen,), dtype=numpy.float)
        xtrim[:len(x)] = x
        
        all = []
        for i in range(q):
            all.append( xtrim[i::q] )
        mysum = numpy.sum( numpy.array(all), axis=0)
        result = numpy.zeros_like(mysum)
        result[:-1] = mysum[:-1]/q
        result[-1] = mysum[-1]/ (q-lendiff)
        return result

def kalman_smooth(orig_rows):
    global printed_dynamics_name

    obs_frames = orig_rows.field('frame')
    fstart, fend = obs_frames[0], obs_frames[-1]
    frames = numpy.arange(fstart,fend+1)
    idx = frames.searchsorted(obs_frames)
    
    x = numpy.empty( frames.shape, dtype=numpy.float )
    y = numpy.empty( frames.shape, dtype=numpy.float )
    z = numpy.empty( frames.shape, dtype=numpy.float )

    x[idx] = orig_rows.field('x')
    y[idx] = orig_rows.field('y')
    z[idx] = orig_rows.field('z')

    # assemble observations (in meters)
    obs = numpy.vstack(( x,y,z )).T

    # initial state guess: postion = observation, other parameters = 0
    ss = 9
    init_x = numpy.zeros( (ss,) )
    init_x[:3] = obs[0,:]

    P_k1=numpy.eye(ss) # initial state error covariance guess
    initial_position_covariance_estimate = 1e-6 # default: initial guess 1mm ( (1e-3)**2 meters)
    initial_acceleration_covariance_estimate = 15 # default: arbitrary initial guess, rather large

    for i in range(0,3):
        P_k1[i,i]=initial_position_covariance_estimate
    for i in range(6,9):
        P_k1[i,i]=initial_acceleration_covariance_estimate

    dynamics_name = 'fly dynamics, high precision calibration, units: mm'
    if not printed_dynamics_name:
        print 'using "%s" for Kalman smoothing'%(dynamics_name,)
        printed_dynamics_name = True
    model = flydra.kalman.dynamic_models.get_dynamic_model_dict()[dynamics_name]
    params = flydra.kalman.params    
    xsmooth, Psmooth = adskalman.kalman_smoother(obs,
                                                 params.A,
                                                 params.C,
                                                 model['Q'],
                                                 model['R'],
                                                 init_x,
                                                 P_k1,
                                                 valid_data_idx=idx)
    return frames, xsmooth, Psmooth

def observations2smoothed(obj_id,orig_rows):
    frames, xsmooth, Psmooth = kalman_smooth(orig_rows)
    obj_id_array = numpy.ones( frames.shape, dtype = numpy.uint32 )*numpy.uint32(obj_id)
    KalmanEstimates = flydra.kalman.flydra_kalman_utils.KalmanEstimates
    field_names = tables.Description(KalmanEstimates().columns)._v_names
    list_of_xhats = [xsmooth[:,0],xsmooth[:,1],xsmooth[:,2],
                     xsmooth[:,3],xsmooth[:,4],xsmooth[:,5],
                     xsmooth[:,6],xsmooth[:,7],xsmooth[:,8],
                     ]
    list_of_Ps = [Psmooth[:,0,0],Psmooth[:,1,1],Psmooth[:,2,2],
                  Psmooth[:,3,3],Psmooth[:,4,4],Psmooth[:,5,5],
                  Psmooth[:,6,6],Psmooth[:,7,7],Psmooth[:,8,8],
                  ]
    timestamps = numpy.zeros( (len(frames),))
    list_of_cols = [obj_id_array,frames,timestamps]+list_of_xhats+list_of_Ps
    assert len(list_of_cols)==len(field_names) # double check that definition didn't change on us
    rows = numpy.rec.fromarrays(list_of_cols,
                                names = field_names)
    return rows

def matfile2rows(data_file,obj_id):
    
    obj_ids = data_file['kalman_obj_id']
    cond = obj_ids == obj_id
    if 0:
        obj_idxs = numpy.nonzero(cond)[0]
        indexer = obj_idxs
    else:
        # benchmarking showed this is 20% faster
        indexer = cond
    
    obj_id_array = data_file['kalman_obj_id'][cond]
    obj_id_array = obj_id_array.astype(numpy.uint32)
    id_frame = data_file['kalman_frame'][cond]
    id_x = data_file['kalman_x'][cond]
    id_y = data_file['kalman_y'][cond]
    id_z = data_file['kalman_z'][cond]
    id_xvel = data_file['kalman_xvel'][cond]
    id_yvel = data_file['kalman_yvel'][cond]
    id_zvel = data_file['kalman_zvel'][cond]
    id_xaccel = data_file['kalman_xaccel'][cond]
    id_yaccel = data_file['kalman_yaccel'][cond]
    id_zaccel = data_file['kalman_xaccel'][cond]

    KalmanEstimates = flydra.kalman.flydra_kalman_utils.KalmanEstimates
    field_names = tables.Description(KalmanEstimates().columns)._v_names
    list_of_xhats = [id_x,id_y,id_z,
                     id_xvel,id_yvel,id_zvel,
                     id_xaccel,id_yaccel,id_zaccel,
                     ]
    z = numpy.nan*numpy.ones(id_x.shape)
    list_of_Ps = [z,z,z,
                  z,z,z,
                  z,z,z,

                  ]
    timestamps = numpy.zeros( (len(frames),))    
    list_of_cols = [obj_id_array,id_frame,timestamps]+list_of_xhats+list_of_Ps
    assert len(list_of_cols)==len(field_names) # double check that definition didn't change on us
    rows = numpy.rec.fromarrays(list_of_cols,
                                names = field_names)
    
    return rows

class LazyRecArrayMimic:
    xtable = {'x':'kalman_x',
              'y':'kalman_y',
              'z':'kalman_z',
              'frame':'kalman_frame',
              'xvel':'kalman_xvel',
              'yvel':'kalman_yvel',
              'zvel':'kalman_zvel',
              }
    def __init__(self,data_file,obj_id):
        self.data_file = data_file
        obj_ids = self.data_file['kalman_obj_id']
        self.cond = obj_ids == obj_id
    def field(self,name):
        return self.data_file[self.xtable[name]][self.cond]
    
class CachingAnalyzer:
    def load_data(self,obj_id,data_file,use_kalman_smoothing=True,
                  frames_per_second=100.0):
        if isinstance(data_file,str):
            if data_file.endswith('_smoothed.mat'):
                data_file = scipy.io.loadmat(data_file)
        
        if isinstance(data_file,dict):
            is_mat_file = True
        else:
            is_mat_file = False
            result_h5_file = data_file
            preloaded_dict = self.loaded_cache.get(result_h5_file,None)
            if preloaded_dict is None:
                preloaded_dict = self._load_dict(result_h5_file)
            kresults = preloaded_dict['kresults']
        
        if is_mat_file:
            if use_kalman_smoothing is not True:
                raise ValueError('use of .mat file requires Kalman smoothing')

            if 0:
                rows = matfile2rows(data_file,obj_id)
            else:
                rows = LazyRecArrayMimic(data_file,obj_id)
        else:
            if not use_kalman_smoothing:
                obj_ids = preloaded_dict['obj_ids']
                idxs = numpy.nonzero(obj_ids == obj_id)[0]
                rows = kresults.root.kalman_estimates.readCoordinates(idxs,flavor='numpy')
            else:
                obs_obj_ids = preloaded_dict['obs_obj_ids']
                obs_idxs = numpy.nonzero(obs_obj_ids == obj_id)[0]
                # Kalman observations are already always in meters, no scale factor needed
                orig_rows = kresults.root.kalman_observations.readCoordinates(obs_idxs,flavor='numpy')
                rows = observations2smoothed(obj_id,orig_rows)  # do Kalman smoothing
                
        return rows

    def get_raw_positions(self,
                          obj_id,
                          data_file,
                          use_kalman_smoothing=True,
                          ):
        """get raw data (Kalman smoothed if data has been pre-smoothed)"""
        rows = self.load_data( obj_id,data_file,use_kalman_smoothing=use_kalman_smoothing)
        xsA = rows.field('x')
        ysA = rows.field('y')
        zsA = rows.field('z')
        
        XA = numpy.vstack((xsA,ysA,zsA)).T
        return XA

    def get_obj_ids(self,data_file):
        if isinstance(data_file,dict):
            is_mat_file = True
        else:
            is_mat_file = False
            result_h5_file = data_file
            preloaded_dict = self.loaded_cache.get(result_h5_file,None)
            if preloaded_dict is None:
                preloaded_dict = self._load_dict(result_h5_file)
        if is_mat_file:
            uoi = numpy.unique(data_file['kalman_obj_id'])
            return uoi
        else:
            preloaded_dict = self.loaded_cache.get(data_file,None)
            if preloaded_dict is None:
                preloaded_dict = self._load_dict(data_file)
            return preloaded_dict['unique_obj_ids']
    
    def calculate_trajectory_metrics(self,
                                     obj_id,
                                     data_file,#result_h5_file or .mat dictionary
                                     use_kalman_smoothing=True,
                                     frames_per_second=100.0,
                                     method='position based',
                                     method_params=None,
                                     hide_first_point=True, # velocity bad there
                                     ):
        """calculate trajectory metrics

        arguments:
        ----------
        obj_id - int, the object id
        data_file - string of pytables filename, the pytables file object, or data dict from .mat file
        frames_per_second - float, framerate of data
        use_kalman_smoothing - boolean, if False, use original, causal Kalman filtered data (rather than Kalman smoothed observations)
        
        method_params for 'position based':
        -----------------------------------
        'downsample' - decimation factor
        
        returns:
        --------
        results - dictionary

        results dictionary always contains:
        -----------------------------------
        
        
        """
        
        rows = self.load_data( obj_id, data_file,use_kalman_smoothing=use_kalman_smoothing)
        numpyerr = numpy.seterr(all='raise')
        try:
            if method_params is None:
                method_params = {}

            RAD2DEG = 180/numpy.pi
            DEG2RAD = 1.0/RAD2DEG

            results = {}

            if method == 'position based':
                ##############
                # load data                
                framesA = rows.field('frame')
                xsA = rows.field('x')

                ysA = rows.field('y')
                zsA = rows.field('z')
                XA = numpy.vstack((xsA,ysA,zsA)).T
                time_A = (framesA - framesA[0])/frames_per_second

                xvelsA = rows.field('xvel')
                yvelsA = rows.field('yvel')
                zvelsA = rows.field('zvel')
                velA = numpy.vstack((xvelsA,yvelsA,zvelsA)).T
                speedA = numpy.sqrt(numpy.sum(velA**2,axis=1))

                ##############
                # downsample
                skip = method_params.get('downsample',3)

                Aindex = numpy.arange(len(framesA))
                AindexB = my_decimate( Aindex, skip )
                AindexB = numpy.round(AindexB).astype(numpy.int)

                xsB = my_decimate(xsA,skip) # time index B - downsampled by 'skip' amount
                ysB = my_decimate(ysA,skip)
                zsB = my_decimate(zsA,skip)
                time_B = my_decimate(time_A,skip)

                XB = numpy.vstack((xsB,ysB,zsB)).T

                ###############################
                # calculate horizontal velocity

                # central difference
                xdiffsF = xsB[2:]-xsB[:-2] # time index F - points B, but inside one point each end
                ydiffsF = ysB[2:]-ysB[:-2]
                zdiffsF = zsB[2:]-zsB[:-2]
                time_F = time_B[1:-1]
                XF = XB[1:-1]
                AindexF = AindexB[1:-1]

                delta_tBC = skip/frames_per_second # delta_t valid for B and C time indices
                delta_tF = 2*delta_tBC # delta_t valid for F time indices

                xvelsF = xdiffsF/delta_tF
                yvelsF = ydiffsF/delta_tF
                zvelsF = zdiffsF/delta_tF

                velsF = numpy.vstack((xvelsF,yvelsF,zvelsF)).T
                speedsF = numpy.sqrt(numpy.sum(velsF**2,axis=1))

                h_speedsF = numpy.sqrt(numpy.sum(velsF[:,:2]**2,axis=1))
                v_speedsF = velsF[:,2]

                headingsF = numpy.arctan2( ydiffsF, xdiffsF )
                #headings2[0] = headings2[1] # first point is invalid

                headingsF_u = numpy.unwrap(headingsF)

                dheadingG_dt = (headingsF_u[2:]-headingsF_u[:-2])/(2*delta_tF) # central difference
                time_G = time_F[1:-1]

                norm_velsF = velsF / speedsF[:,numpy.newaxis] # make norm vectors ( mag(x)=1 )
                if 0:
                    #forward diff
                    time_K = (timeF[1:]+timeF[:-1])/2 # same spacing as F, but in between
                    delta_tK = delta_tF
                    cos_angle_diffsK = [] # time base K - between F
                    for i in range(len(norm_velsF)-1):
                        v1 = norm_velsF[i+1]
                        v2 = norm_velsF[i]
                        cos_angle_diff = numpy.dot(v1,v2) # dot product = mag(a) * mag(b) * cos(theta)
                        cos_angle_diffsK.append( cos_angle_diff )
                    angle_diffK = numpy.arccos(cos_angle_diffsK)
                    angular_velK = angle_diffK/delta_tK
                else:
                    #central diff
                    delta_tG = delta_tF
                    cos_angle_diffsG = [] # time base K - between F
                    for i in range(len(norm_velsF)-2):
                        v1 = norm_velsF[i+2]
                        v2 = norm_velsF[i]
                        cos_angle_diff = numpy.dot(v1,v2) # dot product = mag(a) * mag(b) * cos(theta)
                        cos_angle_diffsG.append( cos_angle_diff )
                    angle_diffG = numpy.arccos(cos_angle_diffsG)
                    angular_velG = angle_diffG/(2*delta_tG)

    ##            times = numpy.arange(0,len(xs))/frames_per_second
    ##            times2 = times[::skip]
    ##            dt_times2 = times2[1:-1]

                if hide_first_point:
                    slicer = slice(1,None,None)
                else:
                    slicer = slice(0,None,None)

                results = {}
                if use_kalman_smoothing:
                    results['kalman_smoothed_rows'] = rows
                results['time_kalmanized'] = time_A[slicer]  # times for position data
                results['X_kalmanized'] = XA[slicer] # raw position data from Kalman (not otherwise downsampled or smoothed)
                results['vel_kalmanized'] = velA[slicer] # raw velocity data from Kalman (not otherwise downsampled or smoothed)
                results['speed_kalmanized'] = speedA[slicer] # raw speed data from Kalman (not otherwise downsampled or smoothed)

                results['time_t'] = time_F[slicer] # times for most data
                results['X_t'] = XF[slicer]
                results['vels_t'] = velsF[slicer] # 3D velocity
                results['speed_t'] = speedsF[slicer]
                results['h_speed_t'] = h_speedsF[slicer]
                results['v_speed_t'] = v_speedsF[slicer]
                results['coarse_heading_t'] = headingsF[slicer] # in 2D plane (note: not body heading)

                results['time_dt'] = time_G # times for angular velocity data
                results['h_ang_vel_dt'] = dheadingG_dt
                results['ang_vel_dt'] = angular_velG
            else:
                raise ValueError('unknown saccade detection algorithm')
        finally:
            numpy.seterr(**numpyerr)
        return results
    
    def get_smoothed(self,
                     obj_id,
                     data_file,#result_h5_file or .mat dictionary
                     frames_per_second=100.0,
                     ):
        rows = self.load_data( obj_id, data_file,
                               use_kalman_smoothing=True,
                               frames_per_second=frames_per_second,
                               )
        results = {}
        results['kalman_smoothed_rows'] = rows
        return results
                        
    def detect_saccades(self,
                        obj_id,
                        data_file,#result_h5_file or .mat dictionary
                        use_kalman_smoothing=True,
                        frames_per_second=100.0,
                        method='position based',
                        method_params=None,
                        ):
        """detect saccades defined as exceeding a threshold in heading angular velocity

        arguments:
        ----------
        obj_id - int, the object id
        data_file - string of pytables filename, the pytables file object, or data dict from .mat file
        use_kalman_smoothing - boolean, if False use original, causal Kalman filtered data (rather than Kalman smoothed observations)

        method_params for 'position based':
        -----------------------------------
        'downsample' - decimation factor
        'threshold angular velocity (rad/s)' - threshold for saccade detection
        'minimum speed' - minimum velocity for detecting a saccade
        'horizontal only' - only use *heading* angular velocity and *horizontal* speed (like Tamerro & Dickinson 2002)

        returns:
        --------
        results - dictionary, see below

        results dictionary contains:
        -----------------------------------
        'indices' - array of ints, indices into h5file at moment of saccade
        'frames' - array of ints, frame numbers of moment of saccade
        'times' - array of floats, seconds since beginning of trace at moment of saccade
        'X' - n by 3 array of floats, 3D position at each saccade
        
        """
        rows = self.load_data( obj_id, data_file,use_kalman_smoothing=use_kalman_smoothing)
        
        if method_params is None:
            method_params = {}
            
        RAD2DEG = 180/numpy.pi
        DEG2RAD = 1.0/RAD2DEG

        results = {}
        
        if method == 'position based':
            ##############
            # load data
            framesA = rows.field('frame') # time index A - original time points
            xsA = rows.field('x')
            ysA = rows.field('y')
            zsA = rows.field('z')
            XA = numpy.vstack((xsA,ysA,zsA)).T

            time_A = (framesA - framesA[0])/frames_per_second
            
            ##############
            # downsample
            skip = method_params.get('downsample',3)

            Aindex = numpy.arange(len(framesA))
            AindexB = my_decimate( Aindex, skip )
            AindexB = numpy.round(AindexB).astype(numpy.int)
            
            xsB = my_decimate(xsA,skip) # time index B - downsampled by 'skip' amount
            ysB = my_decimate(ysA,skip)
            zsB = my_decimate(zsA,skip)
            time_B = my_decimate(time_A,skip)

            ###############################
            # calculate horizontal velocity

            # central difference
            xdiffsF = xsB[2:]-xsB[:-2] # time index F - points B, but inside one point each end
            ydiffsF = ysB[2:]-ysB[:-2]
            zdiffsF = zsB[2:]-zsB[:-2]
            time_F = time_B[1:-1]
            AindexF = AindexB[1:-1]

            delta_tBC = skip/frames_per_second # delta_t valid for B and C time indices
            delta_tF = 2*delta_tBC # delta_t valid for F time indices

            xvelsF = xdiffsF/delta_tF
            yvelsF = ydiffsF/delta_tF
            zvelsF = zdiffsF/delta_tF

##            # forward difference
##            xdiffsC = xsB[1:]-xsB[:-1] # time index C - midpoints between points B, just inside old B endpoints
##            ydiffsC = ysB[1:]-ysB[:-1]
##            zdiffsC = zsB[1:]-zsB[:-1]
##            time_C = (time_B[1:] + time_B[:-1])*0.5
            
##            xvelsC = xdiffsC/delta_tBC
##            yvelsC = ydiffsC/delta_tBC
##            zvelsC = zdiffsC/delta_tBC
            
            horizontal_only = method_params.get('horizontal only',True)

            if horizontal_only:
                ###################
                # calculate heading

##                headingsC = numpy.arctan2( ydiffsC, xdiffsC )
                headingsF = numpy.arctan2( ydiffsF, xdiffsF )

##                headingsC_u = numpy.unwrap(headingsC)
                headingsF_u = numpy.unwrap(headingsF)

##                # central difference of forward difference
##                dheadingD_dt = (headingsC_u[2:]-headingsC_u[:-2])/(2*delta_tBC) # index now the same as C, but starts one later
##                time_D = time_C[1:-1]
                
##                # forward difference of forward difference
##                dheadingE_dt = (headingsC_u[1:]-headingsC_u[:-1])/(delta_tBC) # index now the same as B, but starts one later
##                time_E = (time_C[1:]+time_C[:-1])*0.5

                # central difference of central difference
                dheadingG_dt = (headingsF_u[2:]-headingsF_u[:-2])/(2*delta_tF) # index now the same as F, but starts one later
                time_G = time_F[1:-1]
                
##                # forward difference of central difference
##                dheadingH_dt = (headingsF_u[1:]-headingsF_u[:-1])/(delta_tF) # index now the same as B?, but starts one later
##                time_H = (time_F[1:]+time_F[:-1])*0.5

                if DEBUG:
                    import pylab
                    pylab.figure()
                    pylab.plot( time_D, dheadingD_dt*RAD2DEG, 'k.-', label = 'forward, central')
                    pylab.plot( time_E, dheadingE_dt*RAD2DEG, 'r.-', label = 'forward, forward')
                    pylab.plot( time_G, dheadingG_dt*RAD2DEG, 'g.-', lw = 2, label = 'central, central')
                    pylab.plot( time_H, dheadingH_dt*RAD2DEG, 'b.-', label = 'central, forward')
                    pylab.legend()
                    pylab.xlabel('s')
                    pylab.ylabel('deg/s')
                
            else: # not horizontal only
                
                #central diff
                velsF = numpy.vstack((xvelsF,yvelsF,zvelsF)).T
                speedsF = numpy.sqrt(numpy.sum(velsF**2,axis=1))
                norm_velsF = velsF / speedsF[:,numpy.newaxis] # make norm vectors ( mag(x)=1 )
                delta_tG = delta_tF
                cos_angle_diffsG = [] # time base K - between F
                for i in range(len(norm_velsF)-2):
                    v1 = norm_velsF[i+2]
                    v2 = norm_velsF[i]
                    cos_angle_diff = numpy.dot(v1,v2) # dot product = mag(a) * mag(b) * cos(theta)
                    cos_angle_diffsG.append( cos_angle_diff )
                angle_diffG = numpy.arccos(cos_angle_diffsG)
                angular_velG = angle_diffG/(2*delta_tG)
                
##                vels2 = numpy.vstack((xvels2,yvels2,zvels2)).T
##                speeds2 = numpy.sqrt(numpy.sum(vels2**2,axis=1))
##                norm_vels2 = vels2 / speeds2[:,numpy.newaxis] # make norm vectors ( mag(x)=1 )
##                cos_angle_diffs = [1] # set initial angular vel to 0 (cos(0)=1)
##                for i in range(len(norm_vels2)-1):
##                    v1 = norm_vels2[i+1]
##                    v2 = norm_vels2[i]
##                    cos_angle_diff = numpy.dot(v1,v2) # dot product = mag(a) * mag(b) * cos(theta)
##                    cos_angle_diffs.append( cos_angle_diff )
##                angle_diff = numpy.arccos(cos_angle_diffs)
##                angular_vel = angle_diff/delta_t2

            ###################
            # peak detection
            
            thresh_rad2 = method_params.get('threshold angular velocity (rad/s)',300*DEG2RAD)
            
            if horizontal_only:
                pos_peak_idxsG = find_peaks(dheadingG_dt,thresh_rad2)
                neg_peak_idxsG = find_peaks(-dheadingG_dt,thresh_rad2)
            
                peak_idxsG = pos_peak_idxsG + neg_peak_idxsG

                if DEBUG:
                    import pylab
                    pylab.figure()
                    pylab.plot( dheadingG_dt*RAD2DEG  )
                    pylab.ylabel('heading angular vel (deg/s)')                    
                    for i in peak_idxsG:
                        pylab.axvline( i )
                    pylab.plot( peak_idxsG, dheadingG_dt[peak_idxsG]*RAD2DEG ,'k.')
            else:
                peak_idxsG = find_peaks(angular_velG,thresh_rad2)
                
            orig_idxsG = numpy.array(peak_idxsG,dtype=numpy.int)
            
            ####################
            # make sure peak is at time when velocity exceed minimum threshold
            min_vel = method_params.get('minimum speed',0.04)
            
            orig_idxsF = orig_idxsG+1 # convert G timebase to F
            if horizontal_only:
                h_speedsF = numpy.sqrt(numpy.sum((numpy.vstack((xvelsF,yvelsF)).T)**2,axis=1))
                valid_condF = h_speedsF[orig_idxsF] > min_vel
            else:
                valid_condF = speedsF[orig_idxsF] > min_vel
            valid_idxsF = orig_idxsF[valid_condF]
            
            ####################
            # output parameters
            
            valid_idxsA = AindexF[valid_idxsF]
            #results['indices'] = take_idxs2[valid_idxs] # this seems silly -- how could it be done?
            results['frames'] = framesA[valid_idxsA]
            results['times'] = time_F[valid_idxsF]
            results['X'] = XA[valid_idxsA]
            
            if DEBUG and horizontal_only:
                pylab.figure()
                ax=pylab.subplot(2,1,1)
                pylab.plot( time_G,dheadingG_dt*RAD2DEG )
                pylab.ylabel('heading angular vel (deg/s)')
                for t in results['times']:
                    pylab.axvline(t)
                    
                pylab.plot(time_G[peak_idxsG],dheadingG_dt[peak_idxsG]*RAD2DEG,'ko')
                
                ax=pylab.subplot(2,1,2,sharex=ax)
                pylab.plot( time_F, h_speedsF)
        else:
            raise ValueError('unknown saccade detection algorithm')
        
        return results

    ###################################
    # Implementatation details below
    ###################################
    
    def __init__(self):
        self.loaded_cache = {}
        
    def _load_dict(self,result_h5_file):
        if isinstance(result_h5_file,str) or isinstance(result_h5_file,unicode):
            kresults = tables.openFile(result_h5_file,mode='r')
            self_should_close = True
        else:
            kresults = result_h5_file
            self_should_close = False
        obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')
        obs_obj_ids = kresults.root.kalman_observations.read(field='obj_id',flavor='numpy')
        unique_obj_ids = numpy.unique(obs_obj_ids)
        preloaded_dict = {'kresults':kresults,
                          'self_should_close':self_should_close,
                          'obj_ids':obj_ids,
                          'obs_obj_ids':obs_obj_ids,
                          'unique_obj_ids':unique_obj_ids,
                          }
        self.loaded_cache[result_h5_file] = preloaded_dict
        return preloaded_dict
    
    def close(self):
        for key,preloaded_dict in self.loaded_cache.iteritems():
            if preloaded_dict['self_should_close']:
                preloaded_dict['kresults'].close()
                preloaded_dict['self_should_close'] = False
                
    def __del__(self):
        self.close()
        
if __name__=='__main__':
    ca = CachingAnalyzer()
    results = ca.detect_saccades(2396,'DATA20061208_181556.kalmanized.h5')
    print results['indices']
    print results['times']
    print results['frames']
    print results['X']
    
