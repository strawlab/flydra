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

import pytz, datetime
pacific = pytz.timezone('US/Pacific')

def doit(
         h5_filename=None,
         start=None,
         stop=None,
         ):
    h5 = PT.openFile( h5_filename, mode='r' )
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
    cam_ids = cam_id2camns.keys()
    cam_ids.sort()

    all_data = h5.root.data2d_distorted[:]
    #all_camns = h5.root.data2d_distorted.read(field='camn')
    #all_frames = h5.root.data2d_distorted.read(field='frame')
    for cam_id_enum, cam_id in enumerate( cam_ids ):
        pylab.subplot( len(cam_ids), 1, cam_id_enum+1)
        camns = cam_id2camns[cam_id]
        for camn in camns:
            #this_idx = numpy.nonzero( all_camns==camn )[0]
            this_idx = numpy.nonzero( all_data['camn']==camn )[0]

    h5.close()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--h5", dest="h5_filename", type='string',
                      help=".h5 file with data2d_distorted (REQUIRED)")

    parser.add_option("--start", dest="start", type='int',
                      help="start frame (.h5 frame number reference)")

    parser.add_option("--stop", dest="stop", type='int',
                      help="stop frame (.h5 frame number reference)")

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit(
         h5_filename=options.h5_filename,
         start=options.start,
         stop=options.stop,
         )

if __name__=='__main__':
    main()
