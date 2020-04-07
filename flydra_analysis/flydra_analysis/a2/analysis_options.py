def add_common_options(parser):
    parser.add_option(
        "-k",
        "--kalman",
        dest="kalman_filename",
        type="string",
        help=".h5 file with kalman data and 3D reconstructor",
    )

    parser.add_option(
        "--fps",
        dest="fps",
        type="float",
        help="frames per second (used for Kalman filtering/smoothing)",
    )

    parser.add_option(
        "--disable-kalman-smoothing",
        action="store_false",
        dest="use_kalman_smoothing",
        default=True,
        help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)",
    )

    parser.add_option(
        "--dynamic-model", type="string", dest="dynamic_model", default=None,
    )

    parser.add_option("--max-z", type="float")

    parser.add_option(
        "--start", type="int", help="first frame to plot", metavar="START"
    )

    parser.add_option("--stop", type="int", help="last frame to plot", metavar="STOP")

    parser.add_option("--obj-only", type="string")

    parser.add_option(
        "--stim-xml",
        type="string",
        default=None,
        help="name of XML file with stimulus info",
    )

    parser.add_option("--up-dir", type="string")
