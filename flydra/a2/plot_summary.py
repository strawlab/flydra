from __future__ import division

import matplotlib
matplotlib.use('Agg')
import pylab

import plot_timeseries
import plot_top_view
import analysis_options
from optparse import OptionParser
import os

def doit(options=None):
    fig=pylab.figure(figsize=(10,7.5))
    figtitle = options.kalman_filename.split('.')[0]
    pylab.figtext(0,0,figtitle)

    subplot={}
    subplot['xy']=fig.add_subplot(3,1,1)
    subplot['xz']=fig.add_subplot(3,1,2)#,sharex=subplot['xy'])
    subplot['z']=fig.add_subplot(3,1,3)

    in_fname = options.kalman_filename
    #out_fname = 'summary-' + os.path.splitext(in_fname)[0] + '.png'
    out_fname = os.path.splitext(in_fname)[0] + '.png'

    print 'plotting'
    options.unicolor = True
    options.show_obj_id = False
    options.show_landing = True
    options.show_track_ends = True
    plot_timeseries.plot_timeseries(subplot=subplot,
                                    options=options)

    plot_top_view.plot_top_and_side_views(subplot=subplot,
                                          options=options)

    for key in ['xy','xz']:
        subplot[key].set_frame_on(False)
        subplot[key].set_xticks([])
        subplot[key].set_yticks([])
        subplot[key].set_xlabel('')
        subplot[key].set_ylabel('')

    print 'saving',out_fname

    fig.savefig(out_fname)

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    analysis_options.add_common_options( parser )

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
