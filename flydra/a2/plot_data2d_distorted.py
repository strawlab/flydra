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
         filenames=None,
         start=None,
         stop=None,
         ):
    for filename in filenames:

        fig = pylab.figure()
        pylab.figtext(0,0,filename)

        h5 = PT.openFile( filename, mode='r' )
        camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)
        cam_ids = cam_id2camns.keys()
        if 0:
            print 'removing cam3'
            cam_ids = [cam_id for cam_id in cam_ids if 'cam3' not in cam_id]
        cam_ids.sort()

        all_data = h5.root.data2d_distorted[:]
        ax = None
        for cam_id_enum, cam_id in enumerate( cam_ids ):
            ax = pylab.subplot( len(cam_ids), 1, cam_id_enum+1, sharex=ax)
            camns = cam_id2camns[cam_id]
            for camn in camns:
                this_idx = numpy.nonzero( all_data['camn']==camn )[0]
                data = all_data[this_idx]
                ax.plot( data['frame'], data['x'], 'r.' )
                ax.plot( data['frame'], data['y'], 'g.' )
            pylab.title(cam_id)

        h5.close()

    if len(filenames):
        pylab.show()
    else:
        print 'nothing to do!'

def main():
    usage = '%prog [options] FILE1 [FILE2] ...'

    parser = OptionParser(usage)

    parser.add_option("--start", dest="start", type='int',
                      help="start frame (.h5 frame number reference)")

    parser.add_option("--stop", dest="stop", type='int',
                      help="stop frame (.h5 frame number reference)")

    (options, args) = parser.parse_args()

    doit(
        filenames=args,
        start=options.start,
        stop=options.stop,
        )

if __name__=='__main__':
    main()
