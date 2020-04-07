from __future__ import print_function

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import numpy
import sys, os, time
from optparse import OptionParser
import tables
import matplotlib.mlab as mlab


def convert(
    infilename, outfilename,
):

    results = tables.open_file(infilename, mode="r")
    ra = results.root.textlog[:]
    results.close()
    mlab.rec2csv(ra, outfilename)


def main():
    usage = "%prog FILE [options]"
    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    infilename = args[0]
    outfilename = os.path.splitext(infilename)[0] + ".textlog"
    convert(infilename, outfilename)


if __name__ == "__main__":
    main()
