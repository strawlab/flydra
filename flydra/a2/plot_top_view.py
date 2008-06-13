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

import core_analysis
import flydra.a2.xml_stimulus as xml_stimulus

import pytz, datetime
pacific = pytz.timezone('US/Pacific')

def plot_err( ax, x, mean, err, color=None ):
    ax.plot( x, mean+err, color=color)
    ax.plot( x, mean-err, color=color)

class Frames2Time:
    def __init__(self,frame0,fps):
        self.f0 = frame0
        self.fps = fps
    def __call__(self,farr):
        f = farr-self.f0
        f2  = f/self.fps
        return f2

def plot_top_and_side_views(subplot=None,
                            kalman_filename=None,
                            fps=None,
                            dynamic_model=None,
                            use_kalman_smoothing=None,
                            options=None,
                            ):
    """
    inputs
    ------
    subplot - a dictionary of matplotlib axes instances with keys 'xy' and/or 'xz'
    fps - the framerate of the data
    """
    assert subplot is not None
    assert kalman_filename is not None
    assert options is not None

    start=options.start
    stop=options.stop
    obj_only=options.obj_only

    ca = core_analysis.CachingAnalyzer()

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)

    if not is_mat_file:
        mat_data = None

        if fps is None:
            fps = result_utils.get_fps( data_file, fail_on_error=False )

        if fps is None:
            fps = 100.0
            import warnings
            warnings.warn('Setting fps to default value of %f'%fps)
        reconstructor = reconstruct.Reconstructor(data_file)
    else:
        reconstructor = None

    if dynamic_model is None:
        dynamic_model = extra['dynamic_model_name']
        print 'detected file loaded with dynamic model "%s"'%dynamic_model
        if dynamic_model.startswith('EKF '):
            dynamic_model = dynamic_model[4:]
        print '  for smoothing, will use dynamic model "%s"'%dynamic_model

    subplots = subplot.keys()
    subplots.sort() # ensure consistency across runs

    dt = 1.0/fps

    if obj_only is not None:
        use_obj_ids = [i for i in use_obj_ids if i in obj_only]

    allX = {}
    frame0 = None
    for obj_id in use_obj_ids:
        try:
            kalman_rows = ca.load_data( obj_id, data_file,
                                        use_kalman_smoothing=use_kalman_smoothing,
                                        dynamic_model_name = dynamic_model,
                                        frames_per_second=fps,
                                        )
        except core_analysis.ObjectIDDataError:
            continue
        kobs_rows = ca.load_dynamics_free_MLE_position( obj_id, data_file )

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

        if options.max_z is not None:
            cond = Xz <= options.max_z

            Xx = numpy.ma.masked_where(~cond,Xx)
            Xy = numpy.ma.masked_where(~cond,Xy)
            Xz = numpy.ma.masked_where(~cond,Xz)
            subplot['xz'].axhline( options.max_z )

        line, = subplot['xy'].plot( Xx, Xy, '.', label='obj %d'%obj_id )

        props = dict(color = line.get_color(),
                     linewidth = line.get_linewidth() )
        subplot['xz'].plot( Xx, Xz, '.', label='obj %d'%obj_id, **props )

    if options.stim_xml:
        stim_xml = xml_stimulus.xml_stimulus_from_filename( options.stim_xml )
        if reconstructor is not None:
            stim_xml.verify_reconstructor(reconstructor)
        file_timestamp = data_file.filename[4:19]
        stim_xml.verify_timestamp( file_timestamp )
        def xy_project(X):
            return X[0], X[1]
        def xz_project(X):
            return X[0], X[2]
        stim_xml.plot_stim( subplot['xy'], projection=xy_project )
        stim_xml.plot_stim( subplot['xz'], projection=xz_project )

    subplot['xy'].set_aspect('equal')
    subplot['xz'].set_aspect('equal')

    subplot['xy'].set_xlabel('x (m)')
    subplot['xy'].set_ylabel('y (m)')

    subplot['xz'].set_xlabel('x (m)')
    subplot['xz'].set_ylabel('z (m)')

def doit(kalman_filename=None,
         fps=None,
         use_kalman_smoothing=True,
         dynamic_model = None,
         options = None,
         ):

    if not use_kalman_smoothing:
        if (dynamic_model is not None):
            print >> sys.stderr, 'ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model options (--dynamic-model)'
            sys.exit(1)

    fig = pylab.figure(figsize=(5,8))
    figtitle = kalman_filename.split('.')[0]
    pylab.figtext(0,0,figtitle)

    ax = None
    subplot ={}
    subplots = ['xy','xz']
    for i, name in enumerate(subplots):
        ax = fig.add_subplot(len(subplots),1,i+1,sharex=ax)
        #ax.grid(True)
        subplot[name] = ax

    plot_top_and_side_views(subplot=subplot,
                            kalman_filename=kalman_filename,
                            fps=fps,
                            dynamic_model=dynamic_model,
                            use_kalman_smoothing=use_kalman_smoothing,
                            options=options,
                            )

    pylab.show()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("-k", "--kalman", dest="kalman_filename", type='string',
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

    parser.add_option("--max-z", type="float")

    parser.add_option("--start", type="int",
                      help="first frame to plot",
                      metavar="START")

    parser.add_option("--stop", type="int",
                      help="last frame to plot",
                      metavar="STOP")

    parser.add_option("--obj-only", type="string",
                      dest="obj_only")

    parser.add_option("--stim-xml",
                      type="string",
                      default=None,
                      help="name of XML file with stimulus info",
                      )

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
         options=options,
         )

if __name__=='__main__':
    main()
