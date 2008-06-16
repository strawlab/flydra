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
    subplot={}
    subplot['z']=fig.add_subplot(3,1,1)
    subplot['xy']=fig.add_subplot(3,1,2)
    subplot['xz']=fig.add_subplot(3,1,3)

    in_fname = options.kalman_filename
    out_fname = 'summary-' + os.path.splitext(in_fname)[0] + '.png'

    print 'saving',out_fname
    plot_top_view.plot_top_and_side_views(subplot=subplot,
                                          options=options)
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
