"""
Check a flydra-saved .h5 file for telltale indicators of some flydra problem.
"""

from __future__ import division

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import sys
import tables
from optparse import OptionParser
import numpy

def check_file(h5_filename,debug=False):

    h5 = tables.openFile(h5_filename,mode='r')

    if hasattr(h5.root,'data2d_distorted'):
        print >> sys.stderr, 'has "data2d_distorted" table, checking line components'

        p1 = h5.root.data2d_distorted.read(field='p1')
        ecc= h5.root.data2d_distorted.read(field='eccentricity')

        nan_idx = numpy.nonzero( numpy.isnan(p1) )[0]
        ecc_should_be_nan = ecc[nan_idx]

        if numpy.any(~numpy.isnan(ecc_should_be_nan)):
            print  >> sys.stderr, 'ERROR: p1 is nan, but eccentricity is OK'

        #for row in h5.root.data2d_distorted:
        #    print row['eccentricity'],row['p1']

    h5.close()

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)
    (options, args) = parser.parse_args()
    h5_filename=args[0]
    check_file(h5_filename,debug=True)

if __name__=='__main__':
    main()
