import argparse
import numpy as np
import pandas
import collections
import progressbar
import os

import flydra.a2.core_analysis as core_analysis
import flydra.analysis.result_utils as result_utils
from flydra.a2.tables_tools import openFileSafe
import flydra.reconstruct as reconstruct

class StringWidget(progressbar.Widget):
    def set_string(self,ts):
        self.ts = ts
    def update(self, pbar):
        if hasattr(self,'ts'):
            return self.ts
        else:
            return ''

def calculate_reprojection_errors(h5_filename=None,
                                  output_h5_filename=None,
                                  kalman_filename=None,
                                  start=None,
                                  stop=None,
                                  show_progress=False,
                                  ):
    if os.path.exists( output_h5_filename ):
        raise RuntimeError(
            "will not overwrite old file '%s'"%output_h5_filename)

    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
        kalman_filename)
    R = reconstruct.Reconstructor(kalman_filename)
    ML_estimates_2d_idxs = data_file.root.ML_estimates_2d_idxs[:]

    out = {'camn':[],
           'frame':[],
           'obj_id':[],
           'dist':[],
           }

    with openFileSafe( h5_filename, mode='r' ) as h5:

        fps = result_utils.get_fps( h5, fail_on_error=True )
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

        # associate framenumbers with timestamps using 2d .h5 file
        data2d = h5.root.data2d_distorted[:] # load to RAM
        data2d_idxs = np.arange(len(data2d))
        h5_framenumbers = data2d['frame']
        h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

        if show_progress:
            string_widget = StringWidget()
            objs_per_sec_widget = progressbar.FileTransferSpeed(unit='obj_ids ')
            widgets=[string_widget, objs_per_sec_widget,
                     progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
            pbar=progressbar.ProgressBar(widgets=widgets,maxval=len(use_obj_ids)).start()

        for obj_id_enum,obj_id in enumerate(use_obj_ids):
            if show_progress:
                string_widget.set_string( '[obj_id: % 5d]'%obj_id )
                pbar.update(obj_id_enum)

            obj_3d_rows = ca.load_dynamics_free_MLE_position( obj_id,
                                                              data_file)
            for this_3d_row in obj_3d_rows:
                # iterate over each sample in the current camera
                framenumber = this_3d_row['frame']
                if start is not None:
                    if not framenumber >= start:
                        continue
                if stop is not None:
                    if not framenumber <= stop:
                        continue
                h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)
                if len(h5_2d_row_idxs) == 0:
                    # At the start, there may be 3d data without 2d data.
                    continue

                X3d = this_3d_row['x'], this_3d_row['y'], this_3d_row['z']

                # If there was a 3D ML estimate, there must be 2D data.

                frame2d = data2d[h5_2d_row_idxs]
                frame2d_idxs = data2d_idxs[h5_2d_row_idxs]

                obs_2d_idx = this_3d_row['obs_2d_idx']
                kobs_2d_data = ML_estimates_2d_idxs[int(obs_2d_idx)]

                # Parse VLArray.
                this_camns = kobs_2d_data[0::2]
                this_camn_idxs = kobs_2d_data[1::2]

                # Now, for each camera viewing this object at this
                # frame, extract images.
                for camn, camn_pt_no in zip(this_camns, this_camn_idxs):
                    cam_id = camn2cam_id[camn]

                    # find 2D point corresponding to object
                    cond = ((frame2d['camn']==camn) &
                            (frame2d['frame_pt_idx']==camn_pt_no))
                    idxs = np.nonzero(cond)[0]
                    if len(idxs)==0:
                        #no frame for that camera (start or stop of file)
                        continue
                    elif len(idxs)>1:
                        print "MEGA WARNING MULTIPLE 2D POINTS\n", camn, camn_pt_no,"\n\n"
                        continue

                    idx = idxs[0]

                    frame2d_row = frame2d[idx]
                    x2d_real = frame2d_row['x'], frame2d_row['y']
                    x2d_reproj = R.find2d( cam_id, X3d, distorted = True )
                    dist = np.sqrt(np.sum((x2d_reproj - x2d_real)**2))

                    out['camn'].append(camn)
                    out['frame'].append(framenumber)
                    out['obj_id'].append(obj_id)
                    out['dist'].append(dist)

    # convert to numpy arrays
    for k in out:
        out[k] = np.array( out[k] )
    reprojection = pandas.DataFrame(out)

    # new tables
    camns = []
    cam_ids = []
    for camn in camn2cam_id:
        camns.append(camn)
        cam_ids.append( camn2cam_id[camn] )
    cam_table = {'camn':np.array(camns),
                 'cam_id':np.array(cam_ids),
                 }
    cam_df = pandas.DataFrame(cam_table)

    # save to disk
    store = pandas.HDFStore(output_h5_filename)
    store.append('reprojection', reprojection, data_columns=reprojection.columns)
    store.append('cameras', cam_df)
    store.close()

def main():
    parser = argparse.ArgumentParser(
        description="calculate per-camera, per-frame, per-object reprojection errors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("h5", type=str,
                        help=".h5 file with data2d_distorted")
    parser.add_argument('-k', "--kalman-file",
                        help="file with 3D data (if different that file with data2d_distorted)")
    parser.add_argument("--output-h5", type=str,
                        help="filename for output .h5 file with data2d_distorted")
    parser.add_argument("--start", type=int, default=None,
                        help="frame number to begin analysis on")
    parser.add_argument("--stop", type=int, default=None,
                        help="frame number to end analysis on")
    parser.add_argument('--progress', action='store_true', default=False,
                        help='show progress bar on console')
    args = parser.parse_args()

    if args.kalman_file is None:
        args.kalman_file = args.h5

    if args.output_h5 is None:
        args.output_h5 = args.kalman_file + '.repro_errors.h5'

    calculate_reprojection_errors(h5_filename=args.h5,
                                  kalman_filename=args.kalman_file,
                                  output_h5_filename=args.output_h5,
                                  start=args.start,
                                  stop=args.stop,
                                  show_progress=args.progress,
                                  )

if __name__=='__main__':
    main()
