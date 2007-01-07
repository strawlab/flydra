import numpy

def get_dynamic_model_dict():

    ss = 9 # length of state vector (state size)
    os = 3 # length of observation vector (observation size)

    dynamic_models = {}
    
    # 'hbird2, units: mm':
    # process covariance
    Q = numpy.zeros((ss,ss))
    for i in range(6,9):
        #Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)
        Q[i,i] = 50.0

    # measurement noise covariance matrix
    #R = 1e-6*numpy.eye(os) # (1mm)**2 = (0.001m)**2
    #R = 1e-4*numpy.eye(os) # (10mm)**2 = (0.01m)**2
    R = 8e-4*numpy.eye(os)
    #R = 2.5e-3*numpy.eye(os) # (50mm)**2 = (0.05m)**2

    dynamic_models['hbird2, units: mm'] = dict(
        scale_factor=1e-3,
        #n_sigma_accept=2.8,
        n_sigma_accept=2.4,
        max_variance_dist_meters=0.040, # allow error to grow to 40 mm before dropping
        initial_position_covariance_estimate=1e-4, #10mm ( (1e-2)**2 meters)
        #initial_acceleration_covariance_estimate=15,
        initial_acceleration_covariance_estimate=10,
        Q=Q,
        R=R)
    
    # 'fly dynamics, high precision calibration, units: mm':
    # process covariance
    Q = numpy.zeros((ss,ss))
    for i in range(6,9):
        Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)

    # measurement noise covariance matrix
    R = 1e-6*numpy.eye(os) # (1mm)**2 = (0.001m)**2

    dynamic_models['fly dynamics, high precision calibration, units: mm'] = dict(
        scale_factor=1e-3,
        n_sigma_accept=3.0,
        max_variance_dist_meters=0.010, # allow error to grow to 10 mm before dropping
        initial_position_covariance_estimate=1e-6, #1mm ( (1e-3)**2 meters)
        initial_acceleration_covariance_estimate=15,
        Q=Q,
        R=R)
    
    # 'fly dynamics, low precision calibration, units: mm':
    # process covariance
    Q = numpy.zeros((ss,ss))
    for i in range(6,9):
        Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)

    # measurement noise covariance matrix
    R = 1e-4*numpy.eye(os) # (10mm)**2 = (0.01m)**2

    dynamic_models['fly dynamics, low precision calibration, units: mm'] = dict(
        scale_factor=1e-3,
        n_sigma_accept=3.0,
        max_variance_dist_meters=0.010, # allow error to grow to 10 mm before dropping
        initial_position_covariance_estimate=1e-4, #10mm ( (1e-2)**2 meters)
        initial_acceleration_covariance_estimate=15,
        Q=Q,
        R=R)
            
    # 'hummingbird dynamics, units: mm'

    # process covariance
    Q = numpy.zeros((ss,ss))
    for i in range(6,9):
        Q[i,i] = 100.0 # acceleration noise (near (10*sec**-2)**2)

    # measurement noise covariance matrix
    R = 0.0025*numpy.eye(os) # (5cm)**2 = (0.05m)**2

    dynamic_models['hummingbird dynamics, units: mm'] = dict(
        scale_factor=1e-3, # units: mm
        n_sigma_accept=5.0, # XXX should reduce later?
        max_variance_dist_meters=0.050, # allow error to grow to 50 mm before dropping
        initial_position_covariance_estimate=0.0025, #50mm ( (5e-2)**2 meters)
        initial_acceleration_covariance_estimate=15,
        Q=Q,
        R=R)

    return dynamic_models
