import flydra.reconstruct as reconstruct
from optparse import OptionParser
import xml.etree.ElementTree as ET
import StringIO

def pretty_dump(e, ind=''):
    # from http://www.devx.com/opensource/Article/33153/0/page/4

    # start with indentation
    s = ind
    # put tag (don't close it just yet)
    s += '<' + e.tag
    # add all attributes
    for (name, value) in e.items():
        s += ' ' + name + '=' + "'%s'" % value
    # if there is text close start tag, add the text and add an end tag
    if e.text and e.text.strip():
        s += '>' + e.text + '</' + e.tag + '>'
    else:
        # if there are children...
        if len(e) > 0:
            # close start tag
            s += '>'
            # add every child in its own line indented
            for child in e:
                s += '\n' + pretty_dump(child, ind + '  ')
            # add closing tag in a new line
            s += '\n' + ind + '</' + e.tag + '>'
        else:
            # no text and no children, just close the starting tag
            s += ' />'
    return s

def doit(calsource,options=None):
    r = reconstruct.Reconstructor(calsource)
    if options.scaled:
        r = r.get_scaled()
    root = ET.Element("root")
    r.add_element(root)
    child = root[0]
    if options.pretty:
        result = pretty_dump(child,ind='  ')
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

