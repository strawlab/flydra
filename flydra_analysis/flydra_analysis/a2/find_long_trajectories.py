from __future__ import print_function
from __future__ import absolute_import
import pylab
import numpy
import pickle
import tables as PT


def main(files=None, output_fname=None, fake=False):
    assert files is not None
    assert output_fname is not None

    condition_names = files.keys()

    results = {}
    for col_num, condition_name in enumerate(condition_names):
        filenames = files[condition_name]

        for filename in filenames:
            print("filename", filename)
            kresults = PT.open_file(filename, mode="r")
            obj_ids = kresults.root.kalman_estimates.read(
                field="obj_id", flavor="numpy"
            )
            use_obj_ids = obj_ids

            use_obj_ids = numpy.unique(use_obj_ids)
            print(len(use_obj_ids))

            if fake:
                continue

            objid_by_n_observations = {}
            for obj_id_enum, obj_id in enumerate(use_obj_ids):
                if obj_id_enum % 100 == 0:
                    print("reading %d of %d" % (obj_id_enum, len(use_obj_ids)))
                if PT.__version__ <= "1.3.3":
                    obj_id_find = int(obj_id)
                else:
                    obj_id_find = obj_id

                observation_frame_idxs = kresults.root.ML_estimates.get_where_list(
                    kresults.root.ML_estimates.cols.obj_id == obj_id_find,
                    flavor="numpy",
                )

                observation_frames = kresults.root.ML_estimates.read_coordinates(
                    observation_frame_idxs, field="frame", flavor="numpy"
                )

                n_observations = len(observation_frames)
                objid_by_n_observations.setdefault(n_observations, []).append(obj_id)

            results.setdefault(condition_name, {})[filename] = objid_by_n_observations
            kresults.close()

    if fake:
        return

    fd = open(output_fname, mode="wb")
    pickle.dump(results, fd)
    fd.close()


if __name__ == "__main__":
    from .conditions2 import files

    main(files=files, output_fname="trajectory_lengths2.pkl", fake=False)
