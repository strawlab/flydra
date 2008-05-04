# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20080112

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
import flydra.analysis.result_utils as result_utils

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
        self.f0 = int(frame0)
        self.fps = fps
    def __call__(self,farr):
        farr = numpy.array(farr,dtype=numpy.int64)
        f = farr-self.f0
        f2  = f/self.fps
        return f2

def doit(
         kalman_filename=None,
         fps=None,
         use_kalman_smoothing=True,
         dynamic_model = None,
         start = None,
         stop = None,
         obj_only = None,
         ):

    if not use_kalman_smoothing:
        if (dynamic_model is not None):
            print >> sys.stderr, 'WARNING: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model options (--dynamic-model)'

    ca = core_analysis.CachingAnalyzer()

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)

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

    fig = pylab.figure(figsize=(6,4))
    figtitle=kalman_filename
    pylab.figtext(0,0,figtitle)

    ax = None
    subplot ={}
    subplots = ['x','y','z','vel','accel']
    #subplots = ['x','y','z','vel','accel']
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots),1,i+1,sharex=ax)
        ax.grid(True)
        subplot[name] = ax

    dt = 1.0/fps

    all_vels = []

    if obj_only is not None:
        use_obj_ids = [i for i in use_obj_ids if i in obj_only]

    allX = {}
    frame0 = None

    for obj_id in use_obj_ids:
        try:
            kalman_rows =  ca.load_data( obj_id, data_file,
                                         use_kalman_smoothing=use_kalman_smoothing,
                                         dynamic_model_name = dynamic_model,
                                         frames_per_second=fps,
                                         )
        except core_analysis.ObjectIDDataError:
            continue
        kobs_rows = ca.load_observations( obj_id, data_file )

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
        Xx = kalman_rows['x']
        Xy = kalman_rows['y']
        Xz = kalman_rows['z']

        if frame0 is None:
            frame0 = frame[0]
        f2t = Frames2Time(frame0,fps)

        kws = {}
        if 0:
            if obj_id == 158:
                kws['color'] = .9, .8, 0
            elif obj_id == 160:
                kws['color'] = 0, .45, .70

        line,=subplot['x'].plot( f2t(frame), Xx, label='obj %d'%obj_id, linewidth=2, **kws )
        props = dict(color = line.get_color(),
                     linewidth = line.get_linewidth() )

        subplot['y'].plot( f2t(frame), Xy, label='obj %d'%obj_id, **props )
        subplot['z'].plot( f2t(frame), Xz, label='obj %d'%obj_id, **props )

        X = numpy.array([Xx,Xy,Xz])
        if 0:
            allX[obj_id] = X

        dist_central_diff = (X[:,2:]-X[:,:-2])
        vel_central_diff = dist_central_diff/(2*dt)

        vel2mag = numpy.sqrt(numpy.sum(vel_central_diff**2,axis=0))

        frames2 = frame[1:-1]

        accel4mag = (vel2mag[2:]-vel2mag[:-2])/(2*dt)
        frames4 = frames2[1:-1]

        c = line.get_color()
        subplot['vel'].plot(f2t(frames2), vel2mag, label='obj %d'%obj_id, **props )
        subplot['accel'].plot( f2t(frames4), accel4mag, label='obj %d'%obj_id, **props )
        all_vels.append(vel2mag)
    all_vels = numpy.hstack(all_vels)

    if 1:
        all_vels = all_vels[ all_vels < 2.0 ]
        print 'WARNING: clipping all velocities > 2.0 m/s'
    subplot['x'].set_ylim([0,2])
    #subplot['x'].set_yticks([0,1,2])
    subplot['x'].set_ylabel(r'x ($m$)')

    subplot['y'].set_ylim([0,3])
    #subplot['y'].set_yticks([0,1.5,3])
    subplot['y'].set_ylabel(r'y ($m$)')

    subplot['z'].set_ylim([0,2])
    #subplot['z'].set_yticks([0,1,2])
    subplot['z'].set_ylabel(r'z ($m$)')

    subplot['vel'].set_ylim([0,10])
    #subplot['vel'].set_yticks([0,5,10])
    subplot['vel'].set_ylabel(r'vel ($m/s$)')

    subplot['accel'].set_ylabel(r'acceleration ($m/s^{2}$)')
    #subplot['accel'].set_yticks([-100,0,100])
    subplot['accel'].set_xlabel(r'time ($s$)')

    if 0:
        X1 = allX[158]
        X2 = allX[160]
        dist = numpy.sqrt(numpy.sum((X1-X2)**2,axis=0))
        subplot['dist'].plot( f2t(frame), dist, label='obj %d'%obj_id, **props )

        subplot['dist'].set_ylabel(r'distance ($m$)')

    fig = pylab.figure()
    figtitle=kalman_filename
    pylab.figtext(0,0,figtitle)

    ax = fig.add_subplot(1,1,1)
    bins = numpy.linspace(0,2,30)
    ax.hist(all_vels, bins=bins)
    ax.set_ylabel('counts')
    ax.set_xlabel('velocity (m/s)')
    pylab.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--kalman", dest="kalman_filename", type='string',
                      help=".h5 file with kalman data and 3D reconstructor")

    parser.add_option("--fps", dest='fps', type='float',
                      help="frames per second (used for Kalman filtering/smoothing)")

    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--dynamic-model",
                      type="string",
                      dest="dynamic_model",
                      default=None,
                      )

    parser.add_option("--start", type="int",
                      help="first frame to plot",
                      metavar="START")

    parser.add_option("--stop", type="int",
                      help="last frame to plot",
                      metavar="STOP")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    doit(
         kalman_filename=options.kalman_filename,
         fps = options.fps,
         dynamic_model = options.dynamic_model,
         use_kalman_smoothing=options.use_kalman_smoothing,
         start=options.start,
         stop=options.stop,
         obj_only=options.obj_only,
         )

if __name__=='__main__':
    main()
