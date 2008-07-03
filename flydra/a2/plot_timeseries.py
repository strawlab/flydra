# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20080112

from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

from optparse import OptionParser
import sets, os, sys, math

import pkg_resources
import numpy
import tables as PT
import flydra.reconstruct as reconstruct
import flydra.analysis.result_utils as result_utils
import analysis_options
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.flypos

import matplotlib
rcParams = matplotlib.rcParams
rcParams['xtick.major.pad'] = 10
rcParams['ytick.major.pad'] = 10
import pylab

import flydra.analysis.result_utils as result_utils

import core_analysis

import pytz, datetime
pacific = pytz.timezone('US/Pacific')

def plot_err( ax, x, mean, err, color=None ):
    ax.plot( x, mean+err, color=color)
    ax.plot( x, mean-err, color=color)

class Frames2Time:
    def __init__(self,frame0,fps):
        if not isinstance(frame0,numpy.ma.MaskedArray):
            self.f0 = int(frame0)
        else:
            self.f0 = int(frame0.data)
        self.fps = fps
    def __call__(self,farr):
        farr = numpy.array(farr,dtype=numpy.int64)
        f = farr-self.f0
        f2  = f/self.fps
        return f2

def plot_timeseries(subplot=None,options = None):
    kalman_filename=options.kalman_filename

    if not hasattr(options,'frames'):
        options.frames = False

    if not hasattr(options,'show_landing'):
        options.show_landing = False

    if not hasattr(options,'unicolor'):
        options.unicolor = False

    if not hasattr(options,'show_obj_id'):
        options.show_obj_id = True

    if not hasattr(options,'show_track_ends'):
        options.show_track_ends = False

    start=options.start
    stop=options.stop
    obj_only=options.obj_only
    fps = options.fps
    dynamic_model = options.dynamic_model
    use_kalman_smoothing=options.use_kalman_smoothing

    if not use_kalman_smoothing:
        if (dynamic_model is not None):
            print >> sys.stderr, ('WARNING: disabling Kalman smoothing '
                                  '(--disable-kalman-smoothing) is incompatable '
                                  'with setting dynamic model options (--dynamic-model)')

    ca = core_analysis.get_global_CachingAnalyzer(hack_postmultiply=options.hack_postmultiply)

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)

    do_fuse = False
    if options.stim_xml:
        file_timestamp = data_file.filename[4:19]
        fanout = xml_stimulus.xml_fanout_from_filename( options.stim_xml )
        include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp( timestamp_string=file_timestamp )
        walking_start_stops = fanout.get_walking_start_stops_for_timestamp( timestamp_string=file_timestamp )
        if include_obj_ids is not None:
            use_obj_ids = include_obj_ids
        if exclude_obj_ids is not None:
            use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )
        do_fuse = True
    else:
        walking_start_stops = []


    if dynamic_model is None:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    if not is_mat_file:
        mat_data = None

        if fps is None:
            fps = result_utils.get_fps( data_file, fail_on_error=False )

        if fps is None:
            fps = 100.0
            import warnings
            warnings.warn('Setting fps to default value of %f'%fps)

    dt = 1.0/fps

    all_vels = []

    if obj_only is not None:
        use_obj_ids = [i for i in use_obj_ids if i in obj_only]

    allX = {}
    frame0 = None

    fuse_did_once = False
    for obj_id in use_obj_ids:
        if not do_fuse:
            try:
                kalman_rows =  ca.load_data( obj_id, data_file,
                                             use_kalman_smoothing=use_kalman_smoothing,
                                             dynamic_model_name = dynamic_model,
                                             frames_per_second=fps,
                                             )
            except core_analysis.ObjectIDDataError:
                continue
            #kobs_rows = ca.load_dynamics_free_MLE_position( obj_id, data_file )
        else:
            if fuse_did_once:
                break
            fuse_did_once = True
            kalman_rows = flydra.a2.flypos.fuse_obj_ids(use_obj_ids, data_file,
                                                        dynamic_model_name = dynamic_model,
                                                        frames_per_second=fps)
        frame = kalman_rows['frame']

        if (start is not None) or (stop is not None):
            valid_cond = numpy.ones( frame.shape, dtype=numpy.bool )

            if start is not None:
                valid_cond = valid_cond & (frame >= start)

            if stop is not None:
                valid_cond = valid_cond & (frame <= stop)

            kalman_rows = kalman_rows[valid_cond]
            if not len(kalman_rows):
                continue

            frame = kalman_rows['frame']

        walking_and_flying_kalman_rows = kalman_rows # preserve original data

        for flystate in ['flying','walking']:
            if flystate=='flying':
                # assume flying unless we're told it's walking
                state_cond = numpy.ones( frame.shape, dtype=numpy.bool )
            else:
                state_cond = numpy.zeros( frame.shape, dtype=numpy.bool )

            if len(walking_start_stops):
                frame = walking_and_flying_kalman_rows['frame']

                for walkstart,walkstop in walking_start_stops:
                    frame = walking_and_flying_kalman_rows['frame'] # restore

                    # handle each bout of walking
                    walking_bout = numpy.ones( frame.shape, dtype=numpy.bool)
                    if walkstart is not None:
                        walking_bout &= ( frame >= walkstart )
                    if walkstop is not None:
                        walking_bout &= ( frame <= walkstop )
                    if flystate=='flying':
                        state_cond &= ~walking_bout
                    else:
                        state_cond |= walking_bout

                masked_cond = ~state_cond
                kalman_rows = numpy.ma.masked_where( ~state_cond, walking_and_flying_kalman_rows )
                frame = kalman_rows['frame']

            Xx = kalman_rows['x']
            Xy = kalman_rows['y']
            Xz = kalman_rows['z']

            if frame0 is None:
                frame0 = frame[0]

            if not options.frames:
                f2t = Frames2Time(frame0,fps)
            else:
                def identity(x): return x
                f2t = identity

            kws = {'linewidth':2}
            if options.unicolor:
                kws['color'] = 'k'

            line = None

            if 'x' in subplot:
                line,=subplot['x'].plot( f2t(frame), Xx, label='obj %d (%s)'%(obj_id,flystate),**kws )
                kws['color'] = line.get_color()

            if 'y' in subplot:
                line,=subplot['y'].plot( f2t(frame), Xy, label='obj %d (%s)'%(obj_id,flystate), **kws )
                kws['color'] = line.get_color()

            if 'z' in subplot:
                frame_data = numpy.ma.getdata( frame ) # works if frame is masked or not

                # plot landing time
                if options.show_landing:
                    if flystate == 'flying': # only do this once
                        for walkstart,walkstop in walking_start_stops:
                            if walkstart in frame_data:
                                landing_dix = numpy.nonzero(frame_data==walkstart)[0][0]
                                subplot['z'].plot( [f2t(walkstart)], [Xz.data[landing_dix]], 'rD', ms=10, label='landing' )

                if options.show_track_ends:
                    if flystate == 'flying': # only do this once
                        subplot['z'].plot( f2t([frame_data[0],frame_data[-1]]), [numpy.ma.getdata(Xz)[0],numpy.ma.getdata(Xz)[-1]], 'cd', ms=6, label='track end' )

                line,=subplot['z'].plot( f2t(frame), Xz, label='obj %d (%s)'%(obj_id,flystate), **kws )
                kws['color'] = line.get_color()

                if flystate == 'flying':
                    # only do this once
                    if options.show_obj_id:
                        subplot['z'].text( f2t(frame_data[0]), numpy.ma.getdata(Xz)[0], '%d'%(obj_id,) )

            if numpy.__version__ >= '1.2.0':
                X = numpy.ma.array((Xx,Xy,Xz))
            else:
                # See http://scipy.org/scipy/numpy/ticket/820
                X = numpy.ma.vstack((Xx[numpy.newaxis,:],Xy[numpy.newaxis,:],Xz[numpy.newaxis,:]))

            dist_central_diff = (X[:,2:]-X[:,:-2])
            vel_central_diff = dist_central_diff/(2*dt)

            vel2mag = numpy.ma.sqrt(numpy.ma.sum(vel_central_diff**2,axis=0))
            xy_vel2mag = numpy.ma.sqrt(numpy.ma.sum(vel_central_diff[:2,:]**2,axis=0))

            frames2 = frame[1:-1]

            accel4mag = (vel2mag[2:]-vel2mag[:-2])/(2*dt)
            frames4 = frames2[1:-1]

            if 'vel' in subplot:
                line,=subplot['vel'].plot(f2t(frames2), vel2mag, label='obj %d (%s)'%(obj_id,flystate), **kws )
                kws['color'] = line.get_color()

            if 'xy_vel' in subplot:
                line,=subplot['xy_vel'].plot(f2t(frames2), xy_vel2mag, label='obj %d (%s)'%(obj_id,flystate), **kws )
                kws['color'] = line.get_color()

            if len( accel4mag.compressed()) and 'accel' in subplot:
                line,=subplot['accel'].plot( f2t(frames4), accel4mag, label='obj %d (%s)'%(obj_id,flystate), **kws )
                kws['color'] = line.get_color()

            if flystate=='flying':
                valid_vel2mag = vel2mag.compressed()
                all_vels.append(valid_vel2mag)
    if len(all_vels):
        all_vels = numpy.hstack(all_vels)
    else:
        all_vels = numpy.array([],dtype=float)

    if 1:
        cond = all_vels < 2.0
        if numpy.ma.sum(cond) != len(all_vels):
            all_vels = all_vels[ cond ]
            import warnings
            warnings.warn('clipping all velocities > 2.0 m/s')

    if not options.frames:
        xlabel = 'time ($s$)'
    else:
        xlabel = 'frame'

    if 'x' in subplot:
        subplot['x'].set_ylim([-1,1])
        subplot['x'].set_ylabel(r'x ($m$)')
        subplot['x'].set_xlabel(xlabel)

    if 'y' in subplot:
        subplot['y'].set_ylim([-0.5,1.5])
        subplot['y'].set_ylabel(r'y ($m$)')
        subplot['y'].set_xlabel(xlabel)

    if 'z' in subplot:
        subplot['z'].set_ylim([0,1])
        subplot['z'].set_ylabel(r'z ($m$)')
        subplot['z'].set_xlabel(xlabel)

    if 'vel' in subplot:
        subplot['vel'].set_ylim([0,2])
        subplot['vel'].set_ylabel(r'vel ($m/s$)')
        subplot['vel'].set_xlabel(xlabel)

    if 'xy_vel' in subplot:
        #subplot['xy_vel'].set_ylim([0,2])
        subplot['xy_vel'].set_ylabel(r'horiz vel ($m/s$)')
        subplot['xy_vel'].set_xlabel(xlabel)

    if 'accel' in subplot:
        subplot['accel'].set_ylabel(r'acceleration ($m/s^{2}$)')
        subplot['accel'].set_xlabel(xlabel)

    if 'vel_hist' in subplot:
        ax = subplot['vel_hist']
        bins = numpy.linspace(0,2,50)
        ax.set_title('excluding walking')
        pdf, bins, patches = ax.hist(all_vels, bins=bins, normed=True)
        np = numpy
        ax.set_xlim(0,2)
        ax.set_ylabel('probability density')
        ax.set_xlabel('velocity (m/s)')

def doit(
         options = None,
         ):

    fig = pylab.figure(figsize=(6,4))
    figtitle=options.kalman_filename
    pylab.figtext(0,0,figtitle)

    ax = None
    subplot ={}
    subplots = ['x','y','z','xy_vel','vel','accel']
    #subplots = ['x','y','z','vel','accel']
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots),1,i+1,sharex=ax)
        ax.grid(True)
        subplot[name] = ax


    fig = pylab.figure()
    figtitle=options.kalman_filename
    pylab.figtext(0,0,figtitle)

    ax = fig.add_subplot(1,1,1)
    subplot['vel_hist'] = ax
    ax.grid(True)

    plot_timeseries(subplot=subplot,
                    options = options,
                    )
    pylab.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    analysis_options.add_common_options( parser )

    parser.add_option("--frames", action='store_true',
                      help="plot horizontal axis in frame number (not seconds)",
                      default=False)

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if len(args):
        parser.print_help()
        return

    doit( options=options,
         )

if __name__=='__main__':
    main()
