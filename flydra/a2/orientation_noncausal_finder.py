from __future__ import division
from __future__ import with_statement

import flydra.analysis.result_utils as result_utils
import flydra.a2.core_analysis as core_analysis
import tables
import numpy as np
import flydra.reconstruct
import collections

import matplotlib.pyplot as plt

from image_based_orientation import openFileSafe
import flydra.kalman.ekf as kalman_ekf

ca = core_analysis.get_global_CachingAnalyzer()

slope2modpi = np.arctan # assign function name

if 1:
    from sympy import Symbol, Matrix, sqrt

    tau_rx = 1.0
    tau_ry = 1.0
    tau_rz = 1.0

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    x6 = Symbol('x6')
    x7 = Symbol('x7')


    # from Marins, Yun, Bachmann, McGhee, and Zyda (2001). An Extended
    # Kalman Filter for Quaternion-Based Orientation Estimation Using
    # MARG Sensors. Proceedings of the 2001 IEEE/RSJ International
    # Conference on Intelligent Robots and Systems

    # eqns 9-15
    f1 = -1/tau_rx*x1
    f2 = -1/tau_ry*x2
    f3 = -1/tau_rz*x3
    scale = 2*sqrt(x4**2 + x5**2 + x6**2 + x7**2)
    f4 = 1/scale * ( x3*x5 - x2*x6 + x1*x7 )
    f5 = 1/scale * (-x3*x4 + x1*x6 + x2*x7 )
    f6 = 1/scale * ( x2*x4 - x1*x5 + x3*x7 )
    f7 = 1/scale * (-x1*x4 - x2*x5 + x3*x6 )

    f = (f1,f2,f3,f4,f5,f6,f7)
    f = Matrix(f).T

    #print f
    x = (x1,x2,x3,x4,x5,x6,x7)

    A = f.jacobian(x)
    #print A

    sdict = {x1:1.,
             x2:2.,
             x3:3.,
             x4:4.,
             x5:5.,
             x6:6.,
             x7:7.,
             }


    #print A.subs(sdict)

    At = A.evalf( subs=sdict )
    print At

if 0:
    start = stop = None
    use_obj_ids = [19]

    with openFileSafe( 'DATA20080915_153202.image-based-re2d.kalmanized.h5', mode='r') as kh5:
        with openFileSafe( 'DATA20080915_153202.image-based-re2d.h5', mode='r') as h5:
            reconst = flydra.reconstruct.Reconstructor(kh5)

            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

            # associate framenumbers with timestamps using 2d .h5 file
            data2d = h5.root.data2d_distorted[:] # load to RAM
            data2d_idxs = np.arange(len(data2d))
            h5_framenumbers = data2d['frame']
            h5_frame_qfi = result_utils.QuickFrameIndexer(h5_framenumbers)

            kalman_observations_2d_idxs = kh5.root.kalman_observations_2d_idxs[:]

            for obj_id_enum,obj_id in enumerate(use_obj_ids):
            # Use data association step from kalmanization to load potentially
            # relevant 2D orientations, but discard previous 3D orientation.

                obj_3d_rows = ca.load_dynamics_free_MLE_position( obj_id, kh5)
                slopes_by_camn_by_frame = collections.defaultdict(dict)
                min_frame = np.inf
                max_frame = -np.inf
                for this_3d_row in obj_3d_rows:
                    # iterate over each sample in the current camera
                    framenumber = this_3d_row['frame']
                    if framenumber < min_frame:
                        min_frame = framenumber
                    if framenumber > max_frame:
                        max_frame = framenumber

                    if start is not None:
                        if not framenumber >= start:
                            continue
                    if stop is not None:
                        if not framenumber <= stop:
                            continue
                    h5_2d_row_idxs = h5_frame_qfi.get_frame_idxs(framenumber)

                    frame2d = data2d[h5_2d_row_idxs]
                    frame2d_idxs = data2d_idxs[h5_2d_row_idxs]

                    obs_2d_idx = this_3d_row['obs_2d_idx']
                    kobs_2d_data = kalman_observations_2d_idxs[int(obs_2d_idx)]

                    # Parse VLArray.
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]

                    # Now, for each camera viewing this object at this
                    # frame, extract images.
                    for camn, camn_pt_no in zip(this_camns, this_camn_idxs):
                        # find 2D point corresponding to object
                        cam_id = camn2cam_id[camn]

                        cond = ((frame2d['camn']==camn) &
                                (frame2d['frame_pt_idx']==camn_pt_no))
                        idxs = np.nonzero(cond)[0]
                        assert len(idxs)==1
                        idx = idxs[0]

                        orig_data2d_rownum = frame2d_idxs[idx]
                        frame_timestamp = frame2d[idx]['timestamp']

                        row = frame2d[idx]
                        assert framenumber==row['frame']
                        slopes_by_camn_by_frame[camn][framenumber] = row['slope']

                # now collect in a numpy array for each camn
                frame_range = np.arange(min_frame,max_frame+1)
                slopes_by_camn = {}
                for camn in slopes_by_camn_by_frame.keys():
                    slopes = -1234.5678*np.ones( (len(frame_range),), dtype=np.float)
                    assert np.sum(slopes==-1234.5678)==len(slopes)

                    slopes_by_frame = slopes_by_camn_by_frame[camn]
                    for i,frame in enumerate(frame_range):
                        slopes[i] = slopes_by_frame.get(frame,np.nan)

                    assert np.sum(slopes==-1234.5678)==0
                    #plt.plot(frame_range,slope2modpi(slopes),'.',label=str(camn))

            #plt.legend()
            #plt.show()
            #print 'showed?'
