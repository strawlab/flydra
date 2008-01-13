# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20070112

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
import pylab

import flydra.analysis.result_utils as result_utils

import core_analysis

import pytz, datetime
pacific = pytz.timezone('US/Pacific')

def plot_err( ax, x, mean, err, color=None ):
    ax.plot( x, mean+err, color=color)
    ax.plot( x, mean-err, color=color)

def doit(
         kalman_filename=None,
         fps=None,
         use_kalman_smoothing=True,
         dynamic_model = None,
         start = None,
         stop = None,
         ):

    if not use_kalman_smoothing:
        if (dynamic_model is not None):
            print >> sys.stderr, 'ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model options (--dynamic-model)'
            sys.exit(1)

    ca = core_analysis.CachingAnalyzer()

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)

    fig = pylab.figure(figsize=(18,12))

    ax = None
    subplot ={}
    subplots = ['x','y','z','vel','accel']
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots),1,i+1,sharex=ax)
        subplot[name] = ax

    dt = 1.0/fps

    for obj_id in use_obj_ids:
        try:
            kalman_rows =  ca.load_data( obj_id, data_file,
                                         use_kalman_smoothing=use_kalman_smoothing,
                                         kalman_dynamic_model = dynamic_model,
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

        line,=subplot['x'].plot( frame, Xx, label='obj %d'%obj_id, linewidth=2 )
        props = dict(color = line.get_color(),
                     linewidth = line.get_linewidth() )

        #plot_err( subplot['x'], frame, Xx, kalman_rows['P00'], color=line.get_color() )

        subplot['y'].plot( frame, Xy, label='obj %d'%obj_id, **props )
        subplot['z'].plot( frame, Xz, label='obj %d'%obj_id, **props )

        X = numpy.array([Xx,Xy,Xz])
        vel = numpy.array([kalman_rows['xvel'], kalman_rows['xvel'], kalman_rows['xvel']])
        accel = numpy.array([kalman_rows['xaccel'], kalman_rows['xaccel'], kalman_rows['xaccel']])

        dist_central_diff = (X[:,2:]-X[:,:-2])
        vel_central_diff = dist_central_diff/(2*dt)

        vel2mag = numpy.sqrt(numpy.sum(vel_central_diff**2,axis=0))

        frames2 = frame[1:-1]

        velmag = numpy.sqrt(numpy.sum(vel**2,axis=0))
        accelmag = numpy.sqrt(numpy.sum(accel**2,axis=0))

        subplot['vel'].plot(frame, velmag, label='obj %d'%obj_id, **props )
        c = line.get_color()
        subplot['vel'].plot(frames2, vel2mag, mfc=c, mec=c, color=c, alpha=0.5 )

        subplot['vel'].text( frame[0], velmag[0], '%d'%obj_id )

        subplot['accel'].plot( frame, accelmag, label='obj %d'%obj_id, **props )

    subplot['x'].set_ylim([0,2])
    subplot['x'].set_ylabel('x (m)')

    subplot['y'].set_ylim([0,3])
    subplot['y'].set_ylabel('y (m)')

    subplot['z'].set_ylim([0,2])
    subplot['z'].set_ylabel('z (m)')

    subplot['vel'].set_ylim([0,10])
    subplot['vel'].set_ylabel('|vel| (m/s)')

    subplot['accel'].set_ylabel('|accel| (m/s/s)')

    #subplot['x'].set_xlim([24500,27000])
    #subplot['x'].set_xlim([25700,26000])

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

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit(
         kalman_filename=options.kalman_filename,
         fps = options.fps,
         dynamic_model = options.dynamic_model,
         use_kalman_smoothing=options.use_kalman_smoothing,
         start=options.start,
         stop=options.stop,
         )

if __name__=='__main__':
    main()
