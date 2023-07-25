from __future__ import absolute_import
import pkg_resources

import numpy as np
from .PQmath import (
    ObjectiveFunctionQuats,
    QuatSeq,
    orientation_to_quat,
    quat_to_orient,
    CachingObjectiveFunctionQuats,
    QuatSmoother,
)


class TestPQmath:
    def setUp(self):
        D2R = np.pi / 180.0

        fps = 200.0
        t = np.arange(0, 0.1 * fps) / fps

        yaw_angle = np.sin((2 * np.pi * t) / 4)
        pitch_angle = 45 * D2R * np.ones_like(yaw_angle)

        z_pitch = np.sin(pitch_angle)
        r_pitch = np.cos(pitch_angle)
        direction_vec = np.array(
            [r_pitch * np.cos(yaw_angle), r_pitch * np.sin(yaw_angle), z_pitch]
        ).T

        if 1:
            noise_mag = 0.3
            np.random.seed(4)
            unscaled_noise = np.random.randn(len(t), 3)
            direction_vec += noise_mag * unscaled_noise

        # (re-)normalize

        r = np.sqrt(np.sum(direction_vec ** 2, axis=1))
        direction_vec = direction_vec / r[:, np.newaxis]
        self.Q = QuatSeq([orientation_to_quat(U) for U in direction_vec])

        self.fps = fps

    def test_Qsmooth_slow(self):
        Qsmooth = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            self.Q, objective_func_name="ObjectiveFunctionQuats"
        )

    def test_Qsmooth_caching(self):
        Qsmooth = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            self.Q, objective_func_name="CachingObjectiveFunctionQuats"
        )

    def test_Qsmooth_side_effects(self):
        for of in ["CachingObjectiveFunctionQuats", "ObjectiveFunctionQuats"]:
            yield self.check_Qsmooth_side_effects, of

    def check_Qsmooth_side_effects(self, objective_func_name):
        no_distance_penalty_idxs = [3, 8]

        Qtest = self.Q.copy()
        for i in no_distance_penalty_idxs:
            Qtest[i] = orientation_to_quat((1, 0, 0))
        Qtest_orig = Qtest.copy()
        Qsmooth_v1 = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            Qtest,
            objective_func_name=objective_func_name,
            no_distance_penalty_idxs=no_distance_penalty_idxs,
        )
        # check for side-effects
        for i, (qtest, qexpected) in enumerate(zip(Qtest, Qtest_orig)):
            assert qtest == qexpected

    def test_Qsmooth_missing(self):
        for of in ["CachingObjectiveFunctionQuats", "ObjectiveFunctionQuats"]:
            yield self.check_Qsmooth_missing, of

    def check_Qsmooth_missing(self, objective_func_name):
        no_distance_penalty_idxs = [3, 8]
        objective_func_name = "CachingObjectiveFunctionQuats"
        # objective_func_name='ObjectiveFunctionQuats'

        # If these are really missing and not considered from the
        # distance function, the two results should converge.

        Qtest = self.Q.copy()
        for i in no_distance_penalty_idxs:
            Qtest[i] = orientation_to_quat((1, 0, 0))
        Qsmooth_v1 = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            Qtest,
            objective_func_name=objective_func_name,
            no_distance_penalty_idxs=no_distance_penalty_idxs,
        )

        Qtest = self.Q[:]
        for i in no_distance_penalty_idxs:
            Qtest[i] = orientation_to_quat((0, 0, 1))
        Qsmooth_v2 = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            Qtest,
            objective_func_name=objective_func_name,
            no_distance_penalty_idxs=no_distance_penalty_idxs,
        )
        D2R = np.pi / 180.0
        for i, (q1, q2) in enumerate(zip(Qsmooth_v1, Qsmooth_v2)):

            # Due to early termination criteria, these probably won't
            # be exactly alike, so just compare that angle between two
            # is reasonably small. Could tighten termination criteria
            # and then reduce this threshold angle for errors.

            vec_v1 = quat_to_orient(q1)
            vec_v2 = quat_to_orient(q2)
            dot = np.dot(vec_v1, vec_v2)
            dot = min(1.0, dot)  # clip to prevent arccos returning nan
            angle = np.arccos(dot)
            assert angle < (60 * D2R)

    def test_Qsmooth_both(self):
        Qsmooth_slow = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            self.Q, objective_func_name="ObjectiveFunctionQuats",
        )
        Qsmooth_cache = QuatSmoother(frames_per_second=self.fps).smooth_quats(
            self.Q, objective_func_name="CachingObjectiveFunctionQuats",
        )

        for i, (qs, qc) in enumerate(zip(Qsmooth_slow, Qsmooth_cache)):
            assert qs == qc

