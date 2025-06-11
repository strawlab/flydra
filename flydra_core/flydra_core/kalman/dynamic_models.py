"""linear and non-linear dynamic models for different animals"""

import numpy
import warnings
import re

DEFAULT_MODEL = "mamarama, units: mm"


def _get_decreasing_accel_model(dt=None):
    """get linear dynamical system matrices A and C

    dt is the time-step in seconds
    """
    # distance units are in m
    # time units are in sec
    # thus, velocity is (m/sec)

    ss = 9  # length of state vector (state size)
    os = 3  # length of observation vector (observation size)

    half_dt2 = 0.5 * dt ** 2
    ad = 0.1  # acceleration decay

    # state vector describes a particle in brownian motion
    # [ x y z xvel yvel zvel xaccel yaccel zaccel]

    # process update matrix (time evolution update matrix)
    A = numpy.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0, half_dt2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, half_dt2, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0, half_dt2],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ad, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ad, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ad],
        ]
    )
    A_model_name = "decreasing_accel"

    # measurement prediction matrix
    C = numpy.zeros((os, ss))
    C[:os, :os] = numpy.eye(os)  # directly measure x,y,z positions

    ### process covariance
    ##Q = numpy.zeros((ss,ss))
    ##for i in range(6,9):
    ##    Q[i,i] = 10.0 # acceleration noise (near (3.16m*sec**-2)**2)

    ### measurement noise covariance matrix
    ##R = 1e-6*numpy.eye(os) # (1mm)**2 = (0.001m)**2
    model = {
        "A": A,
        "A_model_name": A_model_name,
        "C": C,
        "ss": ss,
        "os": os,
        "dt": dt,
    }
    return model


def _get_fixed_vel_model(dt=None):
    """get linear dynamical system matrices A and C

    dt is the time-step in seconds
    """
    # distance units are in m
    # time units are in sec
    # thus, velocity is (m/sec)

    ss = 6  # length of state vector (state size)
    os = 3  # length of observation vector (observation size)

    # state vector describes a particle in brownian motion
    # [ x y z xvel yvel zvel]

    # process update matrix (time evolution update matrix)
    A = numpy.array(
        [
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    A_model_name = "fixed_vel"

    # measurement prediction matrix
    C = numpy.zeros((os, ss))
    C[:os, :os] = numpy.eye(os)  # directly measure x,y,z positions

    model = {
        "A": A,
        "A_model_name": A_model_name,
        "C": C,
        "ss": ss,
        "os": os,
        "dt": dt,
    }
    return model


def get_dynamic_model_dict(*args, **kw):
    warnings.warn(
        'DeprecationWarning: call "get_kalman_model()", not old "get_dynamic_model_dict()"'
    )
    return create_dynamic_model_dict(*args, **kw)


def create_dynamic_model_dict(dt=None, disable_warning=False):
    """get linear dynamical system matrices

    dt is the time-step in seconds
    """
    if not disable_warning:
        warnings.warn(
            'using deprecated function "create_dynamic_model_dict()". Use "get_kalman_model()"',
            DeprecationWarning,
            stacklevel=2,
        )
    dynamic_models = {}

    ######################################
    # 'hbird, units: mm':
    # process covariance

    # WARNING: these parameters haven't been tested since the
    # consolidation of the flydra calibration stuff in July-August
    # 2012.

    base_model_dict = _get_fixed_vel_model(dt)
    ss = base_model_dict["ss"]
    os = base_model_dict["os"]

    Q = numpy.zeros((ss, ss))
    for i in range(0, 3):
        Q[i, i] = (0.04) ** 2
    for i in range(3, 6):
        Q[i, i] = (0.4) ** 2

    # measurement noise covariance matrix
    R = 1e-2 * numpy.eye(os)

    newdict = dict(
        # data association parameters
        # birth model
        hypothesis_test_max_acceptable_error=50.0,
        min_dist_to_believe_new_meters=0.2,  # 20 cm
        min_dist_to_believe_new_sigma=3.0,
        initial_position_covariance_estimate=1e-2,
        initial_velocity_covariance_estimate=10,
        # death model
        max_variance_dist_meters=0.08,
        max_frames_skipped=10,
        # kalman filter parameters
        Q=Q,
        R=R,
    )
    newdict.update(base_model_dict)
    dynamic_models["hbird, units: mm"] = newdict

    ######################################

    # 'mamarama, units: mm':
    # process covariance

    base_model_dict = _get_fixed_vel_model(dt)
    ss = base_model_dict["ss"]
    os = base_model_dict["os"]

    if 1:
        # this form is after N. Shimkin's lecture notes in
        # Estimation and Identification in Dynamical Systems
        # http://webee.technion.ac.il/people/shimkin/Estimation09/ch8_target.pdf
        assert ss == 6
        T33 = dt ** 3 / 3.0
        T22 = dt ** 2 / 2.0
        T = dt
        Q = numpy.array(
            [
                [T33, 0, 0, T22, 0, 0],
                [0, T33, 0, 0, T22, 0],
                [0, 0, T33, 0, 0, T22],
                [T22, 0, 0, T, 0, 0],
                [0, T22, 0, 0, T, 0],
                [0, 0, T22, 0, 0, T],
            ]
        )
    # the scale of the covariance
    Q = 0.01 * Q

    # measurement noise covariance matrix
    R = 1e-7 * numpy.eye(os)

    newdict = dict(
        # data association parameters
        # birth model
        hypothesis_test_max_acceptable_error=5.0,
        min_dist_to_believe_new_meters=0.02,
        min_dist_to_believe_new_sigma=10.0,
        initial_position_covariance_estimate=1e-3,
        initial_velocity_covariance_estimate=10,
        # death model
        max_variance_dist_meters=0.25,
        max_frames_skipped=10,
        # kalman filter parameters
        Q=Q,
        R=R,
    )
    newdict.update(base_model_dict)
    dynamic_models["mamarama, units: mm"] = newdict

    ######################################
    # 'fishbowl40':
    # process covariance

    base_model_dict = _get_fixed_vel_model(dt)
    ss = base_model_dict["ss"]
    os = base_model_dict["os"]

    if 1:
        # this form is after N. Shimkin's lecture notes in
        # Estimation and Identification in Dynamical Systems
        # http://webee.technion.ac.il/people/shimkin/Estimation09/ch8_target.pdf
        assert ss == 6
        T33 = dt ** 3 / 3.0
        T22 = dt ** 2 / 2.0
        T = dt
        Q = numpy.array(
            [
                [T33, 0, 0, T22, 0, 0],
                [0, T33, 0, 0, T22, 0],
                [0, 0, T33, 0, 0, T22],
                [T22, 0, 0, T, 0, 0],
                [0, T22, 0, 0, T, 0],
                [0, 0, T22, 0, 0, T],
            ]
        )
    # the scale of the covariance
    Q = 0.01 * Q

    # measurement noise covariance matrix
    R = 1e-7 * numpy.eye(os)

    newdict = dict(
        # data association parameters
        # birth model
        hypothesis_test_max_acceptable_error=5.0,  # Big Fish 5    Smallfish 5
        min_dist_to_believe_new_meters=0.05,  # Big Fish 0.05   Smallfish 0.02
        min_dist_to_believe_new_sigma=10.0,
        initial_position_covariance_estimate=1e-3,
        initial_velocity_covariance_estimate=10,
        # death model
        max_variance_dist_meters=0.125,
        max_frames_skipped=30,
        # kalman filter parameters
        Q=Q,
        R=R,
    )
    newdict.update(base_model_dict)
    dynamic_models["fishbowl40"] = newdict

    ######################################

    # 'hydra, units: m':
    # process covariance

    # WARNING: these parameters haven't been tested since the
    # consolidation of the flydra calibration stuff in July-August
    # 2012.

    base_model_dict = _get_fixed_vel_model(dt)
    ss = base_model_dict["ss"]
    os = base_model_dict["os"]

    Q = numpy.zeros((ss, ss))
    for i in range(0, 3):
        Q[i, i] = (0.01) ** 2
    for i in range(3, 6):
        Q[i, i] = (0.5) ** 2

    Q = Q * 1000 ** 2  # convert to meters

    # measurement noise covariance matrix
    R = 1e-3 * numpy.eye(os)

    newdict = dict(
        # data association parameters
        # birth model
        hypothesis_test_max_acceptable_error=50.0,
        min_dist_to_believe_new_meters=0.08,  # 8 cm
        min_dist_to_believe_new_sigma=3.0,
        initial_position_covariance_estimate=1e-6,
        initial_velocity_covariance_estimate=1,
        # death model
        max_variance_dist_meters=0.08,
        max_frames_skipped=10,
        # kalman filter parameters
        Q=Q,
        R=R,
    )
    newdict.update(base_model_dict)
    dynamic_models["hydra, units: m"] = newdict

    ## ##################################################

    return dynamic_models


class EKFAllParams(dict):
    """abstract base class hold all parameters for data association and EK filtering"""

    def __init__(self):
        self["ss"] = 6
        self["isEKF"] = True


class MamaramaMMEKFAllParams(EKFAllParams):
    """Drosophila non-linear dynamic model for EKF"""

    def __init__(self, dt=None):
        super(MamaramaMMEKFAllParams, self).__init__()
        assert dt is not None
        linear_dict = get_kalman_model(name="mamarama, units: mm", dt=dt)

        # update some parameters from linear model
        for key in [
            "initial_position_covariance_estimate",
            "max_frames_skipped",
            "A",
            "Q",
            "dt",
            "hypothesis_test_max_acceptable_error",
            "min_dist_to_believe_new_meters",
            "min_dist_to_believe_new_sigma",
            "max_variance_dist_meters",
        ]:
            self[key] = linear_dict[key]
        self["ekf_observation_covariance_pixels"] = numpy.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float64
        )
        self[
            "distorted_pixel_euclidian_distance_accept"
        ] = 20.0  # distance in the raw image plane (i.e. before radial undistortion)


class Fishbowl40EKFAllParams(EKFAllParams):
    def __init__(self, dt=None):
        super(Fishbowl40EKFAllParams, self).__init__()
        assert dt is not None
        linear_dict = get_kalman_model(name="fishbowl40", dt=dt)

        # update some parameters from linear model
        for key in [
            "initial_position_covariance_estimate",
            "max_frames_skipped",
            "A",
            "Q",
            "dt",
            "hypothesis_test_max_acceptable_error",
            "min_dist_to_believe_new_meters",
            "min_dist_to_believe_new_sigma",
            "max_variance_dist_meters",
        ]:
            self[key] = linear_dict[key]
        self["ekf_observation_covariance_pixels"] = numpy.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float64
        )
        self[
            "distorted_pixel_euclidian_distance_accept"
        ] = 20.0  # distance in the raw image plane (i.e. before radial undistortion)


class HydraMEKFAllParams(EKFAllParams):
    # WARNING: these parameters haven't been tested since the
    # consolidation of the flydra calibration stuff in July-August
    # 2012.

    def __init__(self, dt=None):
        super(HydraMEKFAllParams, self).__init__()
        assert dt is not None
        linear_dict = get_kalman_model(name="hydra, units: m", dt=dt)

        # update some parameters from linear model
        for key in [
            "initial_position_covariance_estimate",
            "max_frames_skipped",
            "A",
            "Q",
            "dt",
        ]:
            self[key] = linear_dict[key]
        self["ekf_observation_covariance_pixels"] = numpy.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float64
        )

        self["Q"] = self["Q"] / (1000 ** 2)
        self["min_dist_to_believe_new_meters"] = 0.2
        self["min_dist_to_believe_new_sigma"] = 10.0

        self[
            "distorted_pixel_euclidian_distance_accept"
        ] = 20.0  # distance in the raw image plane (i.e. before radial undistortion)

        if 0:
            # restrictive (better for e.g. making new calibration)
            self["max_variance_dist_meters"] = 0.25
        else:
            # loosy-goosy
            self["max_variance_dist_meters"] = 2  # let grow huge


class HbirdEKFAllParams(EKFAllParams):
    # WARNING: these parameters haven't been tested since the
    # consolidation of the flydra calibration stuff in July-August
    # 2012.

    def __init__(self, dt=None):
        super(HbirdEKFAllParams, self).__init__()
        assert dt is not None
        linear_dict = get_kalman_model(name="mamarama, units: mm", dt=dt)

        # update some parameters from linear model
        for key in [
            "initial_position_covariance_estimate",
            "max_frames_skipped",
            "A",
            "Q",
            "dt",
            "hypothesis_test_max_acceptable_error",
            "min_dist_to_believe_new_meters",
            "min_dist_to_believe_new_sigma",
            "max_variance_dist_meters",
        ]:
            self[key] = linear_dict[key]
        self["ekf_observation_covariance_pixels"] = numpy.array(
            [[15.0, 0.0], [0.0, 15.0]], dtype=numpy.float64
        )
        # distance in the raw image plane (i.e. before radial undistortion)
        self["distorted_pixel_euclidian_distance_accept"] = 15.0


ekf_models = {
    "EKF mamarama, units: mm": MamaramaMMEKFAllParams,
    "EKF fishbowl40": Fishbowl40EKFAllParams,
    "EKF hydra, units: m": HydraMEKFAllParams,
    "EKF hbird, units: mm": HbirdEKFAllParams,
}


def get_model_names(ekf_ok=True):
    """get names of available Kalman models"""
    model_dict = create_dynamic_model_dict(dt=0.01, disable_warning=True)
    valid_names = model_dict.keys()
    if ekf_ok:
        valid_names += ekf_models.keys()
    valid_names.sort()
    return valid_names


def get_kalman_model(name=None, dt=None):
    """create a return a Kalman model given a name and timestamp"""
    if name is None:
        raise ValueError("cannot get Kalman model unless name is specified")
    if name.startswith("EKF"):
        if name in ekf_models:
            klass = ekf_models[name]
            kalman_model = klass(dt=dt)
        else:
            raise KeyError("unknown EKF model: %s" % str(name))
    else:
        if name.startswith("fixed_vel_model"):
            # specify fixed_vel_model parameters as string

            # This is modified from Python Library ref 4.2.5:
            floatre = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

            rexp = (
                r"fixed_vel_model\(posQ=%s,velQ=%s,scalarR=%s,init_posV=%s,init_velV=%s\)"
                % (floatre, floatre, floatre, floatre, floatre)
            )
            matchobj = re.match(rexp, name)
            posQ, velQ, scalarR, init_posV, init_velV = map(float, matchobj.groups())

            base_model_dict = _get_fixed_vel_model(dt)
            ss = base_model_dict["ss"]
            os = base_model_dict["os"]

            Q = numpy.zeros((ss, ss))
            for i in range(0, 3):
                Q[i, i] = posQ
            for i in range(3, 6):
                Q[i, i] = velQ

            # measurement noise covariance matrix
            R = scalarR * numpy.eye(os)

            kalman_model = dict(
                Q=Q,
                R=R,
                initial_position_covariance_estimate=init_posV,
                initial_velocity_covariance_estimate=init_velV,
            )
            kalman_model.update(base_model_dict)
            return kalman_model
        model_dict = create_dynamic_model_dict(dt=dt, disable_warning=True)
        try:
            kalman_model = model_dict[name]
        except KeyError:
            valid_names = get_model_names()
            raise KeyError(
                "'%s', valid model names: %s"
                % (str(name), ", ".join(map(repr, valid_names)))
            )
    return kalman_model
