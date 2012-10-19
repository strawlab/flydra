from __future__ import division
from __future__ import with_statement

from orientation_ekf_fitter import *
import analysis_options

def plot_ori(kalman_filename=None,
             h5=None,
             obj_only=None,
             start=None,
             stop=None,
             output_filename=None,
             options=None,
             ):
    if output_filename is not None:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fps = None
    if h5 is not None:
        h5f = tables.openFile(h5,mode='r')
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5f)
        fps = result_utils.get_fps( h5f )
        h5f.close()
    else:
        camn2cam_id = {}

    use_kalman_smoothing=options.use_kalman_smoothing
    fps = options.fps
    dynamic_model = options.dynamic_model

    ca = core_analysis.get_global_CachingAnalyzer()

    (obj_ids, use_obj_ids, is_mat_file, data_file,
     extra) = ca.initial_file_load(kalman_filename)

    if dynamic_model is None:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    with openFileSafe( kalman_filename,
                       mode='r') as kh5:
        if fps is None:
            fps = result_utils.get_fps( kh5, fail_on_error=True )

        kmle = kh5.root.ML_estimates[:] # load into RAM

        if start is not None:
            kmle = kmle[ kmle['frame'] >= start ]

        if stop is not None:
            kmle = kmle[ kmle['frame'] <= stop ]

        all_mle_obj_ids = kmle['obj_id']

        # walk all tables to get all obj_ids
        all_obj_ids = {}
        parent = kh5.root.ori_ekf_qual
        for group in parent._f_iterNodes():
            for table in group._f_iterNodes():
                assert table.name.startswith('obj')
                obj_id = int(table.name[3:])
                all_obj_ids[obj_id] = table

        if obj_only is None:
            use_obj_ids = all_obj_ids.keys()
            mle_use_obj_ids = list(np.unique(all_mle_obj_ids))
            missing_objs = list(set(mle_use_obj_ids) - set(use_obj_ids))
            if len(missing_objs):
                warnings.warn(
                    'orientation not fit for %d obj_ids'%(len(missing_objs),))
            use_obj_ids.sort()
        else:
            use_obj_ids = obj_only

        # now, generate plots
        fig = plt.figure()
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512,sharex=ax1)
        ax3 = fig.add_subplot(513,sharex=ax1)
        ax4 = fig.add_subplot(514,sharex=ax1)
        ax5 = fig.add_subplot(515,sharex=ax1)
        for obj_id in use_obj_ids:
            table = all_obj_ids[obj_id]
            rows = table[:]

            if start is not None:
                rows = rows[ rows['frame'] >= start ]

            if stop is not None:
                rows = rows[ rows['frame'] <= stop ]

            frame=rows['frame']
            # get camns
            camns = []
            for colname in table.colnames:
                if colname.startswith('dist'):
                    camn = int(colname[4:])
                    camns.append(camn)
            for camn in camns:
                label = camn2cam_id.get( camn, 'camn%d'%camn )
                theta = rows['theta%d'%camn]
                used = rows['used%d'%camn]
                dist = rows['dist%d'%camn]
                line,=ax1.plot(frame,theta*R2D,'o',mew=0,ms=2.0,label=label)
                c = line.get_color()
                ax2.plot(frame[used],dist[used]*R2D,'o',color=c,
                         mew=0,label=label)
                ax2.plot(frame[~used],dist[~used]*R2D,'o',color=c,
                         mew=0,ms=2.0)
            # plot 3D orientation
            mle_row_cond = all_mle_obj_ids==obj_id
            rows_this_obj = kmle[mle_row_cond]
            frame = rows_this_obj['frame']
            hz = [rows_this_obj['hz_line%d'%i] for i in range(6)]
            #hz = np.rec.fromarrays(hz,names=['hz%d'%for i in range(6)])
            hz = np.vstack(hz).T
            orient = reconstruct.line_direction(hz)
            ax3.plot(frame,orient[:,0],'ro',mew=0,ms=2.0,label='x')
            ax3.plot(frame,orient[:,1],'go',mew=0,ms=2.0,label='y')
            ax3.plot(frame,orient[:,2],'bo',mew=0,ms=2.0,label='z')

            qual = compute_ori_quality(kh5,rows_this_obj['frame'],obj_id)
            if 1:
                if fps is None:
                    fps = 1.0/200.0

                orinan = np.array(orient,copy=True)
                orinan[ qual < 3.0 ] = np.nan
                sori = ori_smooth(orinan,frames_per_second=fps)
                ax3.plot(frame,sori[:,0],'r-',mew=0,ms=2.0)#,label='x')
                ax3.plot(frame,sori[:,1],'g-',mew=0,ms=2.0)#,label='y')
                ax3.plot(frame,sori[:,2],'b-',mew=0,ms=2.0)#,label='z')

            ax4.plot(frame, qual, 'b-')#, mew=0, ms=3 )

            # --------------
            kalman_rows = ca.load_data( obj_id, kh5,
                                        use_kalman_smoothing=use_kalman_smoothing,
                                        dynamic_model_name = dynamic_model,
                                        return_smoothed_directions = options.smooth_orientations,
                                        frames_per_second=fps,
                                        up_dir=options.up_dir,
                                        min_ori_quality_required=options.ori_qual,
                                        )
            frame = kalman_rows['frame']
            cond = np.ones( frame.shape, dtype=np.bool)
            if options.start is not None:
                cond &= (options.start <= frame)
            if options.stop is not None:
                cond &= (frame <= options.stop)
            kalman_rows=kalman_rows[cond]

            frame = kalman_rows['frame']
            Dx = Dy = Dz = None
            if options.smooth_orientations:
                Dx = kalman_rows['dir_x']
                Dy = kalman_rows['dir_y']
                Dz = kalman_rows['dir_z']
            elif 'rawdir_x' in kalman_rows.dtype.fields:
                Dx = kalman_rows['rawdir_x']
                Dy = kalman_rows['rawdir_y']
                Dz = kalman_rows['rawdir_z']

            ax5.plot( frame, Dx, 'r-', label='dx')
            ax5.plot( frame, Dy, 'g-', label='dy')
            ax5.plot( frame, Dz, 'b-', label='dz')

    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax1.set_ylabel('theta (deg)')
    ax1.legend()

    ax2.set_ylabel('z (deg)')
    ax2.legend()

    ax3.set_ylabel('ori')
    ax3.legend()

    ax4.set_ylabel('quality')

    ax5.set_ylabel('dir')
    ax5.set_xlabel('frame')
    ax5.legend()

    if output_filename is None:
        plt.show()
    else:
        plt.savefig( output_filename )

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)
    analysis_options.add_common_options( parser )
    parser.add_option("--h5", type='string',
                      help=".h5 file with data2d_distorted (REQUIRED)")
    parser.add_option("--output-filename",type="string")
    parser.add_option("--ori-qual", type='float', default=None,
                      help=('minimum orientation quality to use'))
    parser.add_option("--smooth-orientations", action='store_true',
                      help="if displaying orientations, use smoothed data",
                      default=False)
    (options, args) = parser.parse_args()
    if options.kalman_filename is None:
        raise ValueError('--kalman-file option must be specified')
    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)
    plot_ori(kalman_filename=options.kalman_filename,
             h5=options.h5,
             start=options.start,
             stop=options.stop,
             obj_only=options.obj_only,
             output_filename=options.output_filename,
             options=options,
             )

if __name__=='__main__':
    main()
