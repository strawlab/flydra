from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import sets, os, sys, math

import pkg_resources
import numpy
import tables as PT
from optparse import OptionParser
import flydra.reconstruct as reconstruct

import matplotlib
import matplotlib.ticker
import pylab

import flydra.analysis.result_utils as result_utils
import core_analysis

import pytz, datetime

pacific = pytz.timezone('US/Pacific')

all_kalman_lines = {}

def onpick_callback(event):
    # see matplotlib/examples/pick_event_demo.py
    thisline = event.artist
    obj_id = all_kalman_lines[thisline]
    print 'obj_id',obj_id
    if 0:
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print 'picked line:', zip(numpy.take(xdata, ind), numpy.take(ydata, ind))

def doit(
         filenames=None,
         start=None,
         stop=None,
         kalman_filename = None,
         fps=None,
         use_kalman_smoothing=True,
         dynamic_model = None,
         options = None,
         ):

    if not use_kalman_smoothing:
        if (fps is not None) or (dynamic_model is not None):
            print >> sys.stderr, 'ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting fps and dynamic model options (--fps and --dynamic-model)'
            sys.exit(1)

    ax = None
    ax_by_cam = {}
    fig = pylab.figure()

    for filename in filenames:

        figtitle = filename
        if options.obj_only is not None:
            figtitle += ' only showing objects: ' + ' '.join(map(str,options.obj_only))
        pylab.figtext(0,0,figtitle)

        h5 = PT.openFile( filename, mode='r' )

        if fps is None:
            fps = result_utils.get_fps( h5 )

        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        cam_ids = cam_id2camns.keys()
        if 0:
            print 'removing cam3'
            cam_ids = [cam_id for cam_id in cam_ids if 'cam3' not in cam_id]
        cam_ids.sort()

        all_data = h5.root.data2d_distorted[:]
        if start is not None or stop is not None:
            frames = all_data['frame']
            valid_cond = numpy.ones( frames.shape, dtype=numpy.bool)
            if start is not None:
                valid_cond = valid_cond & (frames >= start)
            if stop is not None:
                valid_cond = valid_cond & (frames <= stop)
            all_data = all_data[valid_cond]
            del valid_cond
            del frames

        start_frame = all_data['frame'].min()
        stop_frame = all_data['frame'].max()

        for cam_id_enum, cam_id in enumerate( cam_ids ):
            if cam_id in ax_by_cam:
                ax = ax_by_cam[cam_id]
            else:
                ax = pylab.subplot( len(cam_ids), 1, cam_id_enum+1, sharex=ax)
                ax_by_cam[cam_id] = ax
                ax.fmt_xdata = str
                ax.fmt_ydata = str

            camns = cam_id2camns[cam_id]
            cam_id_n_valid = 0
            for camn in camns:
                this_idx = numpy.nonzero( all_data['camn']==camn )[0]
                data = all_data[this_idx]

                xdata = data['x']
                valid = ~numpy.isnan( xdata )

                data = data[valid]
                del valid

                if options.area_threshold > 0.0:
                    area = data['area']

                    valid2 = area >= options.area_threshold
                    data = data[valid2]
                    del valid2

                n_valid = len( data )
                cam_id_n_valid += n_valid
                if n_valid >= 1:
                    ax.plot( data['frame'], data['x'], 'r.' )
                    ax.plot( data['frame'], data['y'], 'g.' )
            ax.text(0.1,0,'%s: %d pts'%(cam_id,cam_id_n_valid),
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform = ax.transAxes,
                    )
            ax.set_ylabel('%s\npixels'%cam_id)
            ax.set_xlim( (start_frame, stop_frame) )
        ax.set_xlabel('frame')

        h5.close()

    if kalman_filename is not None:
        frame_start = start
        frame_stop = stop

        # copied from save_movies_overlay.py
        ca = core_analysis.CachingAnalyzer()
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)
        if is_mat_file:
            raise ValueError('cannot use .mat file for kalman_filename '
                             'because it is missing the reconstructor '
                             'and ability to get framenumbers')
        R = reconstruct.Reconstructor(data_file)
        R = R.get_scaled( R.get_scale_factor() )

        if options.obj_only is not None:
            use_obj_ids = options.obj_only

        print 'loading frame numbers for kalman objects (estimates)'
        kalman_rows = []
        for obj_id in use_obj_ids:
            my_rows = ca.load_data( obj_id, data_file,
                                    use_kalman_smoothing=use_kalman_smoothing,
                                    kalman_dynamic_model = dynamic_model,
                                    frames_per_second=fps,
                                    )
            kalman_rows.append(my_rows)
        kalman_rows = numpy.concatenate( kalman_rows )
        kalman_3d_frame = kalman_rows['frame']

        if start is not None or stop is not None:
            if start is None:
                start = -numpy.inf
            if stop is None:
                stop = numpy.inf
            valid_cond = (kalman_3d_frame >= start) & (kalman_3d_frame <= stop)

            kalman_rows = kalman_rows[valid_cond]
            kalman_3d_frame = kalman_3d_frame[valid_cond]

        obj_ids = kalman_rows['obj_id']
        use_obj_ids = numpy.unique( obj_ids )
        print 'plotting %d Kalman objects'%(len(use_obj_ids),)
        for obj_id in use_obj_ids:
            cond = obj_ids == obj_id
            x = kalman_rows['x'][cond]
            y = kalman_rows['y'][cond]
            z = kalman_rows['z'][cond]
            w = numpy.ones( x.shape )
            X = numpy.vstack( (x,y,z,w) ).T
            frame = kalman_rows['frame'][cond]
            #print '%d %d %d'%(frame[0],obj_id, len(frame))

            for cam_id in cam_ids:
                ax = ax_by_cam[cam_id]
                x2d = R.find2d(cam_id,X,distorted=True)
                #print '%d %d %s (%f,%f)'%(obj_id,frame[0],cam_id,x2d[0,0],x2d[1,0])
                ax.text( frame[0], x2d[0,0], '%d'%obj_id )
                thisline,=ax.plot( frame, x2d[0,:], 'b-', picker=5) # 5 points tolerance
                all_kalman_lines[thisline] = obj_id
                thisline,=ax.plot( frame, x2d[1,:], 'y-', picker=5) # 5 points tolerance
                all_kalman_lines[thisline] = obj_id
                ax.set_ylim([-100,800])

        if 0:
            # this is forked from flydra_analysis_plot_kalman_2d.py

            # Not finished for now -- we don't need to see what 2D
            # point contributed to the 3D point. Besides,
            # save_movies_overlay.py does that anyway.

            kresults = PT.openFile(kalman_filename,mode='r')
            kobs = kresults.root.kalman_observations
            kframes = kobs.read(field='frame')
            if frame_start is not None:
                k_after_start = numpy.nonzero( kframes>=frame_start )[0]
            else:
                k_after_start = None
            if frame_stop is not None:
                k_before_stop = numpy.nonzero( kframes<=frame_stop )[0]
            else:
                k_before_stop = None

            if k_after_start is not None and k_before_stop is not None:
                k_use_idxs = numpy.intersect1d(k_after_start,k_before_stop)
            elif k_after_start is not None:
                k_use_idxs = k_after_start
            elif k_before_stop is not None:
                k_use_idxs = k_before_stop
            else:
                k_use_idxs = numpy.arange(kobs.nrows)

            obj_ids = kobs.read(field='obj_id')[k_use_idxs]
            obs_2d_idxs = kobs.read(field='obs_2d_idx')[k_use_idxs]
            kframes = kframes[k_use_idxs]

            kobs_2d = kresults.root.kalman_observations_2d_idxs

            # sort data into by-obj-id
            for obj_id in use_obj_ids:
                obs_idxs = numpy.nonzero( obj_ids == obj_id )[0]
                for obs_idx in obs_idxs:
                    camns_and_idxs = kobs_2d[ int(obs_2d_idxs[obs_idx]) ] # cast to int (from numpy scalar type) for pytables
                    if 1:
                        frame = kframes[obs_idx]
                        print 'frame %d, camns_and_idxs: %s'%(frame,str(camns_and_idxs))
                        raise NotImplementedError('') # haven't gotten any further...

    if len(filenames):
        fig.canvas.mpl_connect('pick_event', onpick_callback)
        pylab.show()
    else:
        print 'No filename(s) given -- nothing to do!'

def main():
    usage = '%prog [options] FILE1 [FILE2] ...'

    parser = OptionParser(usage)

    parser.add_option('-k', "--kalman-file", dest="kalman_filename", type='string',
                      help=".h5 file with kalman data and 3D reconstructor")

    parser.add_option("--start", dest="start", type='int',
                      help="start frame (.h5 frame number reference)")

    parser.add_option("--stop", dest="stop", type='int',
                      help="stop frame (.h5 frame number reference)")

    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--fps", dest='fps', type='float',
                      help="frames per second (used for Kalman filtering/smoothing)")

    parser.add_option("--area-threshold", type='float',
                      default = 0.0,
                      help="area of 2D point required for plotting (NOTE: this is not related to the threshold used for Kalmanization)")

    parser.add_option("--dynamic-model",
                      type="string",
                      dest="dynamic_model",
                      default=None,
                      )

    parser.add_option("--obj-only", type="string")

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    doit(
        filenames=args,
        kalman_filename = options.kalman_filename,
        start=options.start,
        stop=options.stop,
        fps = options.fps,
        dynamic_model = options.dynamic_model,
        use_kalman_smoothing=options.use_kalman_smoothing,
        options = options,
        )

if __name__=='__main__':
    main()
