from __future__ import print_function
from __future__ import absolute_import
from . import result_utils
import sys
from optparse import OptionParser


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

    h5_filename = args[0]

    results = result_utils.get_results(h5_filename, create_camera_summary=True)

    print(results.root.data2d_camera_summary.colnames)
    for row in results.root.data2d_camera_summary[:]:
        print(row)

    print()
    print(results.root.cam_info.colnames)
    for row in results.root.cam_info[:]:
        print(row)
    results.close()


if __name__ == "__main__":
    main()
