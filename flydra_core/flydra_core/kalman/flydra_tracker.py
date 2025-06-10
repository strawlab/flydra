from __future__ import print_function
import flydra_core.data_descriptions as data_descriptions
import collections

from flydra_core._flydra_tracked_object import TrackedObject

__all__ = ["TrackedObject", "Tracker"]


class Tracker:
    """
    Handle multiple tracked objects using TrackedObject instances.

    This class keeps a list of objects currently being tracked. It
    also keeps a couple other lists for dealing with cases when the
    tracked objects are no longer 'live'.

    """

    def __init__(
        self,
        reconstructor,
        kalman_model=None,
        save_all_data=False,
        area_threshold=0,
        area_threshold_for_orientation=0.0,
        disable_image_stat_gating=False,
        orientation_consensus=0,
        fake_timestamp=None,
    ):
        """

        arguments
        =========
        reconstructor - reconstructor instance
        kalman_model - dictionary of Kalman filter parameters
        area_threshold - minimum area to consider for tracking use

        """
        self.area_threshold = area_threshold
        self.area_threshold_for_orientation = area_threshold_for_orientation
        self.save_all_data = save_all_data
        self.reconstructor = reconstructor
        self.live_tracked_objects = []
        self.dead_tracked_objects = []  # save for getting data out
        self.kill_tracker_callbacks = []
        self.flushed_callbacks = []
        self.disable_image_stat_gating = disable_image_stat_gating
        self.orientation_consensus = orientation_consensus
        self.fake_timestamp = fake_timestamp
        self.cur_obj_id = 1

        if kalman_model is None:
            raise ValueError("must specify kalman_model")
        self.kalman_model = kalman_model

    def how_many_are_living(self):
        # XXX should we check .kill_me attribute on them?
        return len(self.live_tracked_objects)

    def get_most_recent_data(self):
        results = [tro.get_most_recent_data() for tro in self.live_tracked_objects]
        return results

    def debug_info(level):
        _ = [tro.debug_info(level=level) for tro in self.live_tracked_objects]

    def is_believably_new(self, Xmm, debug=0):
        """Sometimes the Kalman tracker will not gobble all the points
        it should. This still prevents spawning a new Kalman
        tracker."""

        believably_new = True
        X = Xmm
        min_dist_to_believe_new_meters = self.kalman_model[
            "min_dist_to_believe_new_meters"
        ]
        min_dist_to_believe_new_nsigma = self.kalman_model[
            "min_dist_to_believe_new_sigma"
        ]
        results = [tro.get_distance_and_nsigma(X) for tro in self.live_tracked_objects]
        for dist_meters, dist_nsigma in results:
            if debug > 5:
                print("distance in meters, nsigma:", dist_meters, dist_nsigma)
            if (dist_nsigma < min_dist_to_believe_new_nsigma) or (
                dist_meters < min_dist_to_believe_new_meters
            ):
                believably_new = False
                break
        return believably_new

    def remove_duplicate_detections(self, frame, input_data_dict):
        """remove points that are close to current objects being tracked"""

        PT_TUPLE_IDX_FRAME_PT_IDX = data_descriptions.PT_TUPLE_IDX_FRAME_PT_IDX

        (test_frame, bad_camn_pts) = self.last_close_camn_pt_idxs
        assert test_frame == frame

        all_bad_pts = collections.defaultdict(set)
        for camn, ptnum in bad_camn_pts:
            all_bad_pts[camn].add(ptnum)

        output_data_dict = collections.defaultdict(list)
        for camn, camn_list in input_data_dict.items():
            bad_pts = all_bad_pts[camn]
            for element in camn_list:
                pt = element[0]
                ptnum = pt[PT_TUPLE_IDX_FRAME_PT_IDX]
                if ptnum not in bad_pts:
                    output_data_dict[camn].append(element)

        return output_data_dict

    def calculate_a_posteriori_estimates(self, frame, data_dict, camn2cam_id, debug2=0):
        # Allow earlier tracked objects to take all the data they
        # want.

        if debug2 > 1:
            print(self, "gobbling all data for frame %d" % (frame,))

        all_to_gobble = []
        best_by_hash = {}
        to_rewind = []
        # I could parallelize this========================================
        # this is map:
        results = [
            tro.calculate_a_posteriori_estimate(
                frame, data_dict, camn2cam_id, debug1=debug2
            )
            for tro in self.live_tracked_objects
        ]

        # this is reduce:
        all_close_camn_pt_idxs = []
        for idx, result in enumerate(results):
            (used_camns_and_idxs, obs2d_hash, Pmean, close_camn_pt_idxs) = result

            # Two similar lists -- lists of points that will be
            # removed from further consideration. "Gobbling" prevents
            # another object from using it if all the data were in
            # common. Removal of "close", probable duplicate
            # detections, does not remove consideration from
            # pre-existing objects, but will prevent birth of new
            # targets.

            # This works on entire group of 2D
            # observations. (I.e. individual 2D observations can support
            # multiple 3D targets. Only the entire set of 2D
            # observations, given by the obs2d_hash, cannot be shared.)

            all_to_gobble.extend(used_camns_and_idxs)
            all_close_camn_pt_idxs.extend(close_camn_pt_idxs)

            if obs2d_hash is not None:
                if obs2d_hash in best_by_hash:
                    (best_idx, best_Pmean) = best_by_hash[obs2d_hash]
                    if Pmean < best_Pmean:
                        # new value is better than previous best
                        best_by_hash[obs2d_hash] = (idx, Pmean)
                        to_rewind.append(best_idx)
                    else:
                        # old value is still best
                        to_rewind.append(idx)
                else:
                    best_by_hash[obs2d_hash] = (idx, Pmean)
        self.last_close_camn_pt_idxs = (frame, all_close_camn_pt_idxs)

        # End  ================================================================

        if len(all_to_gobble):
            # We deferred gobbling until now - fuse all points to be
            # gobbled and remove them from further consideration.

            # fuse dictionaries
            fused_to_gobble = collections.defaultdict(set)
            for camn, frame_pt_idx, dd_idx in all_to_gobble:
                fused_to_gobble[camn].add(dd_idx)

            # remove data to gobble
            for camn, dd_idx_set in fused_to_gobble.items():
                old_list = data_dict[camn]
                data_dict[camn] = [
                    item for (idx, item) in enumerate(old_list) if idx not in dd_idx_set
                ]

        # Take-back previous observations - starve this Kalman
        # object (which has higher error) so that 2 Kalman objects
        # don't start sharing all observations.
        _ = [
            self.live_tracked_objects[i].remove_previous_observation(debug1=debug2)
            for i in to_rewind
        ]

        # remove tracked objects from live list (when their error grows too large)
        kill_idxs = [
            idx for (idx, tro) in enumerate(self.live_tracked_objects) if tro.kill_me
        ]
        kill_idxs.sort()
        kill_idxs.reverse()
        newly_dead = [self.live_tracked_objects.pop(i) for i in kill_idxs]
        _ = [tro.kill() for tro in newly_dead]
        self.dead_tracked_objects.extend(newly_dead)
        self._flush_dead_queue()
        return data_dict

    def join_new_obj(
        self,
        frame,
        first_observation_orig_units,
        first_observation_Lcoords_orig_units,
        first_observation_camns,
        first_observation_idxs,
        debug=0,
    ):
        obj_id = self.cur_obj_id
        self.cur_obj_id += 1

        tro = TrackedObject(
            self.reconstructor,
            obj_id,
            frame,
            first_observation_orig_units,
            first_observation_Lcoords_orig_units,
            first_observation_camns,
            first_observation_idxs,
            kalman_model=self.kalman_model,
            save_all_data=self.save_all_data,
            area_threshold=self.area_threshold,
            area_threshold_for_orientation=self.area_threshold_for_orientation,
            disable_image_stat_gating=self.disable_image_stat_gating,
            orientation_consensus=self.orientation_consensus,
            fake_timestamp=self.fake_timestamp,
        )
        self.live_tracked_objects.append(tro)

    def kill_all_trackers(self):
        _ = [tro.kill() for tro in self.live_tracked_objects]
        while len(self.live_tracked_objects):
            self.dead_tracked_objects.append(self.live_tracked_objects.pop())
        self._flush_dead_queue()

    def set_killed_tracker_callback(self, callback):
        self.kill_tracker_callbacks.append(callback)

    def set_flushed_callback(self, callback):
        self.flushed_callbacks.append(callback)

    def clear_flushed_callbacks(self):
        self.flushed_callbacks = []

    def _flush_dead_queue(self):
        while len(self.dead_tracked_objects):
            tro = self.dead_tracked_objects.pop(0)
            for callback in self.kill_tracker_callbacks:
                callback(tro)
        for callback in self.flushed_callbacks:
            callback()


def test_cython():
    """Test that cython is working."""
    import flydra_core._flydra_tracked_object as fto
    import numpy as np

    print("Cython module loaded successfully.")
    arr = np.zeros((10,), dtype=np.uint16)
    hashable = fto.obs2d_hashable(arr)
    hash(hashable)  # should not raise an error
