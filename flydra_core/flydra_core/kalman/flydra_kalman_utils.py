import tables as PT
import numpy as np

import flydra_core.data_descriptions
import flydra_core.kalman.dynamic_models
from flydra_core.kalman.point_prob import some_rough_negative_log_likelihood

PT_TUPLE_IDX_X = flydra_core.data_descriptions.PT_TUPLE_IDX_X
PT_TUPLE_IDX_Y = flydra_core.data_descriptions.PT_TUPLE_IDX_Y
PT_TUPLE_IDX_AREA = flydra_core.data_descriptions.PT_TUPLE_IDX_AREA
PT_TUPLE_IDX_SLOPE = flydra_core.data_descriptions.PT_TUPLE_IDX_SLOPE
PT_TUPLE_IDX_ECCENTRICITY = flydra_core.data_descriptions.PT_TUPLE_IDX_ECCENTRICITY
PT_TUPLE_IDX_P1 = flydra_core.data_descriptions.PT_TUPLE_IDX_P1
PT_TUPLE_IDX_P2 = flydra_core.data_descriptions.PT_TUPLE_IDX_P2
PT_TUPLE_IDX_P3 = flydra_core.data_descriptions.PT_TUPLE_IDX_P3
PT_TUPLE_IDX_P4 = flydra_core.data_descriptions.PT_TUPLE_IDX_P4

PT_TUPLE_IDX_FRAME_PT_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX
PT_TUPLE_IDX_CUR_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_CUR_VAL_IDX
PT_TUPLE_IDX_MEAN_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_MEAN_VAL_IDX
PT_TUPLE_IDX_SUMSQF_VAL_IDX = flydra_core.data_descriptions.PT_TUPLE_IDX_SUMSQF_VAL_IDX


class KalmanEstimates(PT.IsDescription):
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.Float64Col(pos=2)  # time of reconstruction
    x = PT.Float32Col(pos=3)
    y = PT.Float32Col(pos=4)
    z = PT.Float32Col(pos=5)
    xvel = PT.Float32Col(pos=6)
    yvel = PT.Float32Col(pos=7)
    zvel = PT.Float32Col(pos=8)
    xaccel = PT.Float32Col(pos=9)
    yaccel = PT.Float32Col(pos=10)
    zaccel = PT.Float32Col(pos=11)
    # save diagonal of P matrix
    P00 = PT.Float32Col(pos=12)
    P11 = PT.Float32Col(pos=13)
    P22 = PT.Float32Col(pos=14)
    P33 = PT.Float32Col(pos=15)
    P44 = PT.Float32Col(pos=16)
    P55 = PT.Float32Col(pos=17)
    P66 = PT.Float32Col(pos=18)
    P77 = PT.Float32Col(pos=19)
    P88 = PT.Float32Col(pos=20)


class KalmanEstimatesVelOnly(PT.IsDescription):
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.Float64Col(pos=2)  # time of reconstruction
    x = PT.Float32Col(pos=3)
    y = PT.Float32Col(pos=4)
    z = PT.Float32Col(pos=5)
    xvel = PT.Float32Col(pos=6)
    yvel = PT.Float32Col(pos=7)
    zvel = PT.Float32Col(pos=8)
    # save diagonal of P matrix
    P00 = PT.Float32Col(pos=9)
    P11 = PT.Float32Col(pos=10)
    P22 = PT.Float32Col(pos=11)
    P33 = PT.Float32Col(pos=12)
    P44 = PT.Float32Col(pos=13)
    P55 = PT.Float32Col(pos=14)


class KalmanEstimatesVelOnlyWithDirection(PT.IsDescription):
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.Float64Col(pos=2)  # time of reconstruction
    x = PT.Float32Col(pos=3)
    y = PT.Float32Col(pos=4)
    z = PT.Float32Col(pos=5)
    xvel = PT.Float32Col(pos=6)
    yvel = PT.Float32Col(pos=7)
    zvel = PT.Float32Col(pos=8)
    # save diagonal of P matrix
    P00 = PT.Float32Col(pos=9)
    P11 = PT.Float32Col(pos=10)
    P22 = PT.Float32Col(pos=11)
    P33 = PT.Float32Col(pos=12)
    P44 = PT.Float32Col(pos=13)
    P55 = PT.Float32Col(pos=14)
    # save estimated direction of long body axis
    rawdir_x = PT.Float32Col(pos=15)
    rawdir_y = PT.Float32Col(pos=16)
    rawdir_z = PT.Float32Col(pos=17)
    dir_x = PT.Float32Col(pos=18)
    dir_y = PT.Float32Col(pos=19)
    dir_z = PT.Float32Col(pos=20)


class KalmanEstimatesVelOnlyPositionCovariance(PT.IsDescription):
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.Float64Col(pos=2)  # time of reconstruction
    x = PT.Float32Col(pos=3)
    y = PT.Float32Col(pos=4)
    z = PT.Float32Col(pos=5)
    xvel = PT.Float32Col(pos=6)
    yvel = PT.Float32Col(pos=7)
    zvel = PT.Float32Col(pos=8)
    # save parts of P matrix
    P00 = PT.Float32Col(pos=9)
    P01 = PT.Float32Col(pos=10)
    P02 = PT.Float32Col(pos=11)
    P11 = PT.Float32Col(pos=12)
    P12 = PT.Float32Col(pos=13)
    P22 = PT.Float32Col(pos=14)
    P33 = PT.Float32Col(pos=15)
    P44 = PT.Float32Col(pos=16)
    P55 = PT.Float32Col(pos=27)


class KalmanEstimatesVelOnlyWithDirectionPositionCovariance(PT.IsDescription):
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    timestamp = PT.Float64Col(pos=2)  # time of reconstruction
    x = PT.Float32Col(pos=3)
    y = PT.Float32Col(pos=4)
    z = PT.Float32Col(pos=5)
    xvel = PT.Float32Col(pos=6)
    yvel = PT.Float32Col(pos=7)
    zvel = PT.Float32Col(pos=8)
    # save parts of P matrix
    P00 = PT.Float32Col(pos=9)
    P01 = PT.Float32Col(pos=10)
    P02 = PT.Float32Col(pos=11)
    P11 = PT.Float32Col(pos=12)
    P12 = PT.Float32Col(pos=13)
    P22 = PT.Float32Col(pos=14)
    P33 = PT.Float32Col(pos=15)
    P44 = PT.Float32Col(pos=16)
    P55 = PT.Float32Col(pos=17)
    # save estimated direction of long body axis
    rawdir_x = PT.Float32Col(pos=18)
    rawdir_y = PT.Float32Col(pos=19)
    rawdir_z = PT.Float32Col(pos=20)
    dir_x = PT.Float32Col(pos=21)
    dir_y = PT.Float32Col(pos=22)
    dir_z = PT.Float32Col(pos=23)


class KalmanSaveInfo(object):
    def __init__(
        self,
        name=None,
        allocate_space_for_direction=False,
        save_covariance="position"  # or 'full' or 'diag'
        # 'position' means to save full covariance for upper 3x3, diag for rest
    ):
        if save_covariance == "full":
            raise NotImplementedError("")
        assert save_covariance in ["position", "diag"]
        self.save_covariance = save_covariance
        model = flydra_core.kalman.dynamic_models.get_kalman_model(name=name, dt=1.0)
        ss = model["ss"]
        if ss == 6:
            if self.save_covariance == "diag":
                if allocate_space_for_direction:
                    self.description = KalmanEstimatesVelOnlyWithDirection
                else:
                    self.description = KalmanEstimatesVelOnly
            else:
                # 'position'
                if allocate_space_for_direction:
                    self.description = (
                        KalmanEstimatesVelOnlyWithDirectionPositionCovariance
                    )
                else:
                    self.description = KalmanEstimatesVelOnlyPositionCovariance
                # indices for upper part of covariance matrix (which is symmetric, so this is sufficient)
                self.rows = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5])
                self.cols = np.array([0, 1, 2, 1, 2, 2, 3, 4, 5])
        elif ss == 9:
            assert self.save_covariance == "diag"
            if allocate_space_for_direction:
                raise NotImplementedError("")
            else:
                self.desription = KalmanEstimates

    def get_description(self):
        return self.description

    def get_save_covariance(self):
        return self.save_covariance

    def covar_mat_to_covar_entries(self, M):
        if self.save_covariance == "diag":
            return np.diag(M).tolist()
        else:
            return np.asarray(M)[self.rows, self.cols].tolist()

    def covar_mats_to_covar_entries(self, Ms):
        result = []
        for (row, col) in zip(self.rows, self.cols):
            result.append(Ms[:, row, col])
        return result


class FilteredObservations(
    PT.IsDescription
):  # Not really "observations" but ML estimates
    obj_id = PT.UInt32Col(pos=0)
    frame = PT.Int64Col(pos=1)
    x = PT.Float32Col(pos=2)
    y = PT.Float32Col(pos=3)
    z = PT.Float32Col(pos=4)
    obs_2d_idx = PT.UInt64Col(pos=5)  # index into VLArray 'ML_estimates_2d_idxs'
    hz_line0 = PT.Float32Col(pos=6)
    hz_line1 = PT.Float32Col(pos=7)
    hz_line2 = PT.Float32Col(pos=8)
    hz_line3 = PT.Float32Col(pos=9)
    hz_line4 = PT.Float32Col(pos=10)
    hz_line5 = PT.Float32Col(pos=11)


ML_estimates_2d_idxs_type = PT.UInt16Atom


def convert_format(current_data, camn2cam_id, area_threshold=0.0, only_likely=False):
    """convert data from format used for Kalman tracker to hypothesis tester"""
    found_data_dict = {}
    first_idx_by_cam_id = {}
    for camn, stuff_list in current_data.items():
        if not len(stuff_list):
            # no data for this camera, continue
            continue
        for (pt_undistorted, projected_line) in stuff_list:
            if not np.isnan(pt_undistorted[0]):  # only use if point was found

                # perform area filtering
                area = pt_undistorted[PT_TUPLE_IDX_AREA]
                if area < area_threshold:
                    continue

                cam_id = camn2cam_id[camn]

                if only_likely:
                    # a quick gating based on image attributes.

                    pt_area = pt_undistorted[PT_TUPLE_IDX_AREA]
                    cur_val = pt_undistorted[PT_TUPLE_IDX_CUR_VAL_IDX]
                    mean_val = pt_undistorted[PT_TUPLE_IDX_MEAN_VAL_IDX]
                    sumsqf_val = pt_undistorted[PT_TUPLE_IDX_SUMSQF_VAL_IDX]

                    p_y_x = some_rough_negative_log_likelihood(
                        pt_area, cur_val, mean_val, sumsqf_val
                    )  # this could even depend on 3d geometry
                    if not np.isfinite(p_y_x):
                        continue

                found_data_dict[cam_id] = pt_undistorted[:9]
                first_idx_by_cam_id[cam_id] = pt_undistorted[PT_TUPLE_IDX_FRAME_PT_IDX]
                break  # algorithm only accepts 1 point per camera
    return found_data_dict, first_idx_by_cam_id
