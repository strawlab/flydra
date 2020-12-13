from __future__ import print_function
import flydra_core.reconstruct as reconstruct
from optparse import OptionParser
import xml.etree.ElementTree as ET

def doit(calsource, options=None):
    r = reconstruct.Reconstructor(calsource)
    if options.scaled:
        r = r.get_scaled()
    root = ET.Element("root")
    r.add_element(root)
    child = root[0]
    result = reconstruct.pretty_dump(child, ind="  ")
    if options.dest:
        with open(options.dest, "w") as the_file:
            the_file.write(result)
        print("saved calibration to %s" % options.dest)
    else:
        print(result)


def main():
    usage = "%prog CALSOURCE [options]"

    parser = OptionParser(usage)

    parser.add_option("--scaled", action="store_true", default=False)

    parser.add_option(
        "--dest", type="string", help="file to save calibration to (e.g.test.xml)"
    )

    (options, args) = parser.parse_args()
    calsource = args[0]
    doit(calsource, options=options)
