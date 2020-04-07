from __future__ import print_function
import flydra_analysis.a2.core_analysis as core_analysis
import numpy as np
import flydra_analysis.a2.utils as utils


class EquivalentObjectFinder:
    def __init__(self, src_h5, dst_h5):
        self.ca = core_analysis.get_global_CachingAnalyzer()
        self.src_h5 = src_h5
        self.dst_h5 = dst_h5
        self.ca.initial_file_load(self.src_h5)

        tmp = self.ca.initial_file_load(self.dst_h5)
        obj_ids, unique_obj_ids, is_mat_file, data_file, extra = tmp

        self.dst_frames = extra["frames"]
        self.dst_obj_ids = obj_ids
        self.dst_unique_obj_ids = unique_obj_ids
        self.ff = utils.FastFinder(self.dst_frames)

    def find_equiv(self, src_obj_id, mean_distance_maximum=None):
        """find the obj_id in the dst file that corresponds to src_obj_id

        arguments
        ---------
        src_obj_id : int
            The obj_id of the object in src_h5 to find.
        mean_distance_maximum : float or None
            The maximum average distance between points in dst and src.

        returns
        -------
        dst_obj_id : int
            The obj_id in dst_h5 that corresponds to the src_obj_id
        """
        # get information from source to identify trace in dest
        src_rows = self.ca.load_data(
            src_obj_id, self.src_h5, use_kalman_smoothing=False
        )
        src_frame = src_rows["frame"]

        if len(src_frame) < 2:
            raise ValueError(
                "Can only find equivalent obj_id if " "2 or more frames present"
            )

        src_X = np.vstack((src_rows["x"], src_rows["y"], src_rows["z"]))
        src_timestamp = src_rows["timestamp"]

        candidate_obj_id = set()
        for f in src_frame:
            idxs = self.ff.get_idxs_of_equal(f)
            for obj_id in self.dst_obj_ids[idxs]:
                candidate_obj_id.add(obj_id)

        candidate_obj_id = list(candidate_obj_id)
        ## print 'candidate_obj_id',candidate_obj_id
        error = []
        for obj_id in candidate_obj_id:
            # get array for each candidation obj_id in destination
            dst_rows = self.ca.load_data(
                obj_id, self.dst_h5, use_kalman_smoothing=False
            )
            dst_frame = dst_rows["frame"]
            dst_X = np.vstack((dst_rows["x"], dst_rows["y"], dst_rows["z"]))
            dst_ff = utils.FastFinder(dst_frame)

            # get indices into destination array for each frame of source
            dst_idxs = dst_ff.get_idx_of_equal(src_frame, missing_ok=1)

            assert len(dst_idxs) == len(src_frame)

            missing_cond = dst_idxs == -1  # these points are in source but not dest
            n_missing = np.sum(missing_cond)
            n_total = len(src_frame)
            present_cond = ~missing_cond

            final_dst_idxs = dst_idxs[present_cond]
            final_src_idxs = np.arange(len(src_frame))[present_cond]

            src_X_i = src_X[:, final_src_idxs]
            dst_X_i = dst_X[:, final_dst_idxs]

            diff = src_X_i - dst_X_i
            dist = np.sqrt(np.sum(diff ** 2, axis=0))
            av_dist = np.mean(dist)

            frac_missing = n_missing / float(n_total)  # 0 = none missing, 1 = all
            ## print 'candidate dst obj_id %d: %s dist, %s missing'%(
            ##     obj_id, av_dist, frac_missing)
            if frac_missing > 0.1:
                this_error = np.inf
            else:
                this_error = av_dist
            error.append(this_error)
        idx = np.argmin(error)
        best_error = error[idx]
        if not np.isfinite(best_error):
            return None  # could not find answer
        else:
            if (mean_distance_maximum is None) or (best_error <= mean_distance_maximum):
                return candidate_obj_id[idx]
            else:
                return None


if __name__ == "__main__":
    finder = EquivalentObjectFinder(
        "DATA20090301_200059.h5", "DATA20090301_200059.kalmanized.h5"
    )
    obj_id = finder.find_equiv(13, mean_distance_maximum=1e-4)
    print("obj_id", obj_id)
