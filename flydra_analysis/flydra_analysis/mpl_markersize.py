from optparse import OptionParser
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def main():
    usage = "%prog [options]"

    parser = OptionParser(usage)
    parser.add_option(
        "--save-fig",
        type="string",
        default=None,
        help="path name of figure to save (exits script " "immediately after save)",
    )
    (options, args) = parser.parse_args()

    x = np.arange(10)
    for msize in [0.5, 1.0, 2.5, 5.0]:
        y = msize * x
        plt.plot(x, y, ".", label="markersize %s" % msize, markersize=msize)
    plt.legend()
    if options.save_fig is not None:
        plt.savefig(options.save_fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
