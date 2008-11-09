import flydra.reconstruct as reconstruct
from optparse import OptionParser
import xml.etree.ElementTree as ET
import StringIO

def doit(calsource,options=None):
    r = reconstruct.Reconstructor(calsource)
    if options.scaled:
        r = r.get_scaled()
    root = ET.Element("root")
    r.add_element(root)
    child = root[0]
    if options.pretty:
        result = reconstruct.pretty_dump(child,ind='  ')
    else:
        tree = ET.ElementTree(child)
        fd = StringIO.StringIO()
        tree.write(fd)
        result = fd.getvalue()
    print result

def main():
    usage = '%prog CALSOURCE [options]'

    parser = OptionParser(usage)

    parser.add_option("--pretty", action='store_true',
                      default=False)

    parser.add_option("--scaled", action='store_true',
                      default=False)

    (options, args) = parser.parse_args()
    calsource = args[0]
    doit(calsource,options=options)

