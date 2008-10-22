import flydra.a2.xml_stimulus as xml_stimulus
from optparse import OptionParser
import subprocess

def doit(stim_xml=None,options=None):
    fanout = xml_stimulus.xml_fanout_from_filename( stim_xml )
    for single_episode in fanout.iterate_single_episodes():
        kh5_files = []
        for child in single_episode:
            if child.tag == 'kh5_file':
                kh5_file = child.attrib['name']
                kh5_files.append( kh5_file )
        if len(kh5_files) != 1:
            raise ValueError('expected one and only one kh5 file in a single_episode')
        kh5_file = kh5_files[0]
        cmd = ['flydra_analysis_plot_summary','-k',kh5_file,'--stim-xml',stim_xml]
        print ' '.join(cmd)
        subprocess.check_call(cmd)

def main():
    usage = '%prog FANOUT_XML_FILENAME [options]'

    parser = OptionParser(usage)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_usage()
        return 1

    stim_xml = args[0]

    doit( options=options,
          stim_xml = stim_xml,
         )

if __name__=='__main__':
    main()
