from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import matplotlib

from . import plot_timeseries
from . import plot_top_view
from . import analysis_options
from . import core_analysis
from optparse import OptionParser
import os


def doit(options=None):
    import pylab  # do after matplotlib.use() call

    fig = pylab.figure(figsize=(10, 7.5))
    figtitle = options.kalman_filename.split(".")[0]
    pylab.figtext(0, 0, figtitle)

    subplot = {}
    subplot["xy"] = fig.add_axes((0.05, 0.55, 0.9, 0.45))
    subplot["xz"] = fig.add_axes((0.05, 0.35, 0.9, 0.2))  # ,sharex=subplot['xy'])

    subplot["z"] = fig.add_axes((0.05, 0.1, 0.8, 0.2))
    subplot["z_hist"] = fig.add_axes((0.85, 0.1, 0.1, 0.2), sharey=subplot["z"])

    in_fname = options.kalman_filename
    # out_fname = 'summary-' + os.path.splitext(in_fname)[0] + '.png'
    out_fname = os.path.splitext(in_fname)[0] + ".png"

    print("plotting")
    options.unicolor = True
    options.show_obj_id = False
    options.show_landing = True
    options.show_track_ends = True
    options.smooth_orientations = False
    plot_timeseries.plot_timeseries(subplot=subplot, options=options)

    plot_top_view.plot_top_and_side_views(subplot=subplot, options=options)

    for key in ["xy", "xz"]:
        subplot[key].set_frame_on(False)
        subplot[key].set_xticks([])
        subplot[key].set_yticks([])
        subplot[key].set_xlabel("")
        subplot[key].set_ylabel("")

    print("saving", out_fname)

    fig.savefig(out_fname)
    if options.interactive:
        pylab.show()


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)

    analysis_options.add_common_options(parser)

    parser.add_option("--interactive", action="store_true", default=False)

    parser.add_option(
        "--fuse",
        action="store_true",
        help="fuse object ids corresponding to a single fly (requires stim-xml fanout)",
        default=False,
    )

    (options, args) = parser.parse_args()

    if options.obj_only is not None:
        options.obj_only = core_analysis.parse_seq(options.obj_only)

    if not options.interactive:
        matplotlib.use("Agg")

    if len(args):
        parser.print_help()
        return

    if options.up_dir is not None:
        options.up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        options.up_dir = None

    doit(options=options,)


if __name__ == "__main__":
    main()
