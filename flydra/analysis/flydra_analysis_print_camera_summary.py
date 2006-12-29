import result_utils
import sys
from optparse import OptionParser

def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)

    parser.add_option("--writeable", action='store_true',dest='writeable',
                      help="open file in writeable mode")

    (options, args) = parser.parse_args()
    
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return

    h5_filename=args[0]
    if options.writeable:
        mode = 'r+'
    else:
        mode = 'r'

    results = result_utils.get_results( h5_filename, mode=mode)
    if not hasattr(results.root,'data2d_camera_summary'):
        print >> sys.stderr, "ERROR: no table 'data2d_camera_summary' (try opening in writeable mode)"
        sys.exit(1)

    print results.root.data2d_camera_summary.colnames    
    for row in results.root.data2d_camera_summary:
        print row
    results.close()

if __name__=='__main__':
    main()
