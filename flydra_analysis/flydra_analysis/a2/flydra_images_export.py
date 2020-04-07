from __future__ import print_function
import tables
import numpy
import scipy
from optparse import OptionParser
import scipy.misc


def doit(filename=None):
    results = tables.open_file(filename, mode="r")
    image_table = results.root.images

    for row in results.root.cam_info:
        cam_id = row["cam_id"]

        arr = getattr(image_table, cam_id)
        image = arr.read()
        if image.ndim == 3 and image.shape[2] == 1:
            # only a single "color" channel
            image = image[:, :, 0]  # drop a dimension (3D->2D)
        mean_luminance = numpy.mean(image.flat)
        print(
            "%s: %dx%d, mean luminance %.1f"
            % (cam_id, image.shape[1], image.shape[0], mean_luminance)
        )
        scipy.misc.imsave("%s.png" % (cam_id,), image)
    results.close()


def main():
    usage = "%prog FILE [options]"

    # A man page can be generated with:
    # 'help2man -N -n %prog %prog > %prog.1'
    parser = OptionParser(usage)

    parser.add_option(
        "--version",
        action="store_true",
        dest="version",
        help="print version and quit",
        default=False,
    )

    (options, args) = parser.parse_args()

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]

    if options.version:
        print("%s %s" % (sys.argv[0], flydra_analysis.version.__version__,))

    doit(filename=h5_filename)


if __name__ == "__main__":
    main()
