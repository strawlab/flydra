import numpy
import math
import warnings

def _get_decreasing_accel_model(dt=None):
    """get linear dynamical system matrices A and C

    dt is the time-step in seconds
    """
    # distance units are in m
    # time units are in sec
    # thus, velocity is (m/sec)

    ss = 9 # length of state vector (state size)
    os = 3 # length of observation vector (observation size)

    half_dt2 = 0.5*dt**2
    ad = 0.1 # acceleration decay

    # state vector describes a particle in brownian motion
    # [ x y z xvel yvel zvel xaccel yaccel zaccel]

    # process update matrix (time evolution update matrix)
    A = numpy.array([[   1. ,    0. ,    0. ,   dt  ,    0. ,    0. , half_dt2 ,   0.     ,    0. ],
                     [   0. ,    1. ,    0. ,    0. ,   dt  ,    0. ,   0.     , half_dt2 ,    0. ],
                     [   0. ,    0. ,    1. ,    0. ,    0. ,   dt  ,   0.     ,   0.     ,  half_dt2  ],
                     [   0. ,    0. ,    0. ,    1. ,    0. ,    0. ,  dt      ,   0.     ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    1. ,    0. ,   0.     ,  dt      ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    0. ,    1. ,   0.     ,   0.     ,   dt  ],
                     [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,   ad     ,   0.     ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,   0.     ,   ad     ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    0. ,    0. ,   0.     ,   0.     ,    ad]])
    A_model_name = 'decreasing_accel'

    # measurement prediction matrix
    C = numpy.zeros((os,ss))
    C[:os,:os] = numpy.eye(os) # directly measure x,y,z positions

    ### process covariance
    ##Q = numpy.zeros((ss,ss))
    ##for i in range(6,9):
    ##    Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)

    ### measurement noise covariance matrix
    ##R = 1e-6*numpy.eye(os) # (1mm)**2 = (0.001m)**2
    model = {'A':A,
             'A_model_name':A_model_name,
             'C':C,
             'ss':ss,
             'os':os,
             'dt':dt,
             }
    return model

def _get_fixed_vel_model(dt=None):
    """get linear dynamical system matrices A and C

    dt is the time-step in seconds
    """
    # distance units are in m
    # time units are in sec
    # thus, velocity is (m/sec)

    ss = 6 # length of state vector (state size)
    os = 3 # length of observation vector (observation size)

    # state vector describes a particle in brownian motion
    # [ x y z xvel yvel zvel]

    # process update matrix (time evolution update matrix)
    A = numpy.array([[   1. ,    0. ,    0. ,   dt  ,    0. ,    0. ],
                     [   0. ,    1. ,    0. ,    0. ,   dt  ,    0. ],
                     [   0. ,    0. ,    1. ,    0. ,    0. ,   dt  ],
                     [   0. ,    0. ,    0. ,    1. ,    0. ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    1. ,    0. ],
                     [   0. ,    0. ,    0. ,    0. ,    0. ,    1. ]])
    A_model_name = 'fixed_vel'

    # measurement prediction matrix
    C = numpy.zeros((os,ss))
    C[:os,:os] = numpy.eye(os) # directly measure x,y,z positions

    model = {'A':A,
             'A_model_name':A_model_name,
             'C':C,
             'ss':ss,
             'os':os,
             'dt':dt,
             }
    return model

def get_dynamic_model_dict(*args,**kw):
    warnings.warn('DeprecationWarning: call "get_kalman_model()", not old "get_dynamic_model_dict()"')
    return create_dynamic_model_dict(*args,**kw)

def create_dynamic_model_dict(dt=None,disable_warning=False):
    """get linear dynamical system matrices

    dt is the time-step in seconds
    """
    if not disable_warning:
        warnings.warn('using deprecated function "create_dynamic_model_dict()". Use "get_kalman_model()"',DeprecationWarning,stacklevel=2)
    dynamic_models = {}

    ######################################
    # 'hbird3, units: mm':
    # process covariance
    base_model_dict = _get_decreasing_accel_model(dt)
    ss = base_model_dict['ss']
    os = base_model_dict['os']

    Q = numpy.zeros((ss,ss))
    for i in range(0,3):
        #Q[i,i] = (0.005)**2
        Q[i,i] = (0.010)**2

    for i in range(3,6):
        Q[i,i] = (.5)**2 # velocity noise

    for i in range(6,9):
        Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)
        #Q[i,i] = 50.0

    # measurement noise covariance matrix
    #R = 1e-6*numpy.eye(os) # (1mm)**2 = (0.001m)**2
    #R = 1e-4*numpy.eye(os) # (10mm)**2 = (0.01m)**2
    #R = 2e-4*numpy.eye(os)
    R = 2.5e-3*numpy.eye(os) # (50mm)**2 = (0.05m)**2
    #R = 2.5e-2*numpy.eye(os)

    newdict = dict(
        # these 2 values are old and could probably be improved:
        min_dist_to_believe_new_meters=0.0,
        min_dist_to_believe_new_sigma=9.0,

        n_sigma_accept=0.5,
        max_variance_dist_meters=math.sqrt(0.06),
        initial_position_covariance_estimate=(0.1)**2, # 30mm2
        #initial_acceleration_covariance_estimate=15,
        initial_velocity_covariance_estimate=50,
        initial_acceleration_covariance_estimate=150,
        max_frames_skipped=25,
        Q=Q,
        R=R)
    newdict.update(base_model_dict)
    dynamic_models['hummingbird dynamics, units: mm'] = newdict
    ######################################

    # 'fly dynamics, high precision calibration, units: mm':
    # process covariance
    base_model_dict = _get_decreasing_accel_model(dt)
    ss = base_model_dict['ss']
    os = base_model_dict['os']

    Q = numpy.zeros((ss,ss))
    for i in range(0,3):
        Q[i,i] = (0.001)**2
    for i in range(3,6):
        Q[i,i] = (0.2)**2
    for i in range(6,9):
        Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)

    # measurement noise covariance matrix
    R = 1e-6*numpy.eye(os)

    newdict = dict(
        min_dist_to_believe_new_meters=0.01, # 1 cm
        min_dist_to_believe_new_sigma=3.0,
        n_sigma_accept=1.0,
        max_variance_dist_meters=0.02,
        initial_position_covariance_estimate=1e-6,
        initial_acceleration_covariance_estimate=15,
        max_frames_skipped=25,
        Q=Q,
        R=R)
    newdict.update(base_model_dict)
    dynamic_models['fly dynamics, high precision calibration, units: mm'] = newdict

    ## ##################################################

    ######################################

    # 'mamarama, units: mm':
    # process covariance

    base_model_dict = _get_fixed_vel_model(dt)
    ss = base_model_dict['ss']
    os = base_model_dict['os']

    Q = numpy.zeros((ss,ss))
    for i in range(0,3):
        Q[i,i] = (0.01)**2
    for i in range(3,6):
        Q[i,i] = (0.5)**2

    # measurement noise covariance matrix
    R = 1e-3*numpy.eye(os)

    newdict = dict(

        # data association parameters

        # birth model
        min_dist_to_believe_new_meters=0.08, # 8 cm
        min_dist_to_believe_new_sigma=3.0,

        initial_position_covariance_estimate=1e-6,
        initial_velocity_covariance_estimate=1,

        # support existint object
        n_sigma_accept=20.0, # geometric euclidian distance

        # death model
        max_variance_dist_meters=0.08,
        max_frames_skipped=10,

        # kalman filter parameters
        Q=Q,
        R=R)
    newdict.update(base_model_dict)
    dynamic_models['mamarama, units: mm'] = newdict

    ## ##################################################

    return dynamic_models

class EKFAllParams(dict):
    """abstract base class hold all parameters for data association and EK filtering"""
    def __init__(self):
        self['ss'] = 6
        self['isEKF']=True

class MamaramaMMEKFAllParams(EKFAllParams):
    def __init__(self,dt=None):
        super( MamaramaMMEKFAllParams, self).__init__()
        assert dt is not None
        linear_dict = get_kalman_model( name='mamarama, units: mm',
                                        dt=dt )

        # update some parameters from linear model
        for key in [#'min_dist_to_believe_new_meters',
                    #'min_dist_to_believe_new_sigma',
                    'initial_position_covariance_estimate',
                    'max_frames_skipped',
                    'A',
                    'Q',
                    'dt',
                    ]:
            self[key] = linear_dict[key]
        self['ekf_observation_covariance_pixels'] = numpy.array( [[1.0, 0.0],
                                                                  [0.0, 1.0]],
                                                                 dtype=numpy.float64 )
        self['min_dist_to_believe_new_meters']=0.2
        self['min_dist_to_believe_new_sigma']=10.0

        self['undistorted_pixel_euclidian_distance_accept']=20.0

        if 1:
            # restrictive (better for e.g. making new calibration)
            self['max_variance_dist_meters']=0.25
            self['n_sigma_accept']=10
        else:
            # loosy-goosy
            self['max_variance_dist_meters']=2 # let grow huge
            self['n_sigma_accept']=40

ekf_models = {'EKF mamarama, units: mm':MamaramaMMEKFAllParams,
              }

def get_model_names(ekf_ok=True):
    model_dict = create_dynamic_model_dict(dt=0.01,disable_warning=True)
    valid_names = model_dict.keys()
    if ekf_ok:
        valid_names += ekf_models.keys()
    valid_names.sort()
    return valid_names

def get_kalman_model( name=None, dt=None ):
    if name is None:
        raise ValueError('cannot get Kalman model unless name is specified')
    if name.startswith('EKF'):
        if name in ekf_models:
            klass = ekf_models[name]
            kalman_model = klass(dt=dt)
        else:
            raise KeyError('unknown EKF model: %s'%str(name))
    else:
        model_dict = create_dynamic_model_dict(dt=dt,disable_warning=True)
        try:
            kalman_model = model_dict[name]
        except KeyError, err:
            valid_names = get_model_names()
            raise KeyError("'%s', valid model names: %s"%(str(name),', '.join(map(repr,valid_names))))
    return kalman_model


