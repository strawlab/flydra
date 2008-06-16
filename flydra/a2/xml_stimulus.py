import xml.etree.ElementTree as ET
import flydra.reconstruct as reconstruct
import flydra.a2.experiment_layout as experiment_layout
import numpy
import os
import hashlib
from core_analysis import parse_seq

class Stimulus(object):

    def __init__(self,root):
        assert root.tag == 'stimxml'
        assert root.attrib['version']=='1'
        self.root = root
        self._R = None

    def _get_reconstructor(self):
        if self._R is None:
            r_node = self.root.find("multi_camera_reconstructor")
            self._R = reconstruct.Reconstructor_from_xml(r_node)
        return self._R

    def verify_reconstructor(self,other_R):
        R = self._get_reconstructor()
        assert isinstance(other_R,reconstruct.Reconstructor)
        assert R == other_R

    def verify_timestamp(self,timestamp):
        timestamp_in_file = False
        for child in self.root:
            if child.tag == 'valid_h5_times':
                valid_times = child.text.split()
                for vt in valid_times:
                    if timestamp == vt.strip():
                        timestamp_in_file = True
                        break
            if timestamp_in_file == True:
                break
        assert timestamp_in_file

    def _get_info_for_cylindrical_arena(self,child):
        assert child.tag == 'cylindrical_arena'
        info = {}
        for v in child:
            if v.tag == 'origin':
                info['origin'] = numpy.array(map(float,v.text.split()))
            elif v.tag == 'axis':
                info['axis'] = numpy.array(map(float,v.text.split()))
            elif v.tag == 'diameter':
                info['diameter'] = float(v.text)
            elif v.tag == 'height':
                info['height'] = float(v.text)
            else:
                raise ValueError('unknown tag: %s'%v.tag)
        return info

    def _get_info_for_cylindrical_post(self,child):
        assert child.tag == 'cylindrical_post'
        verts = []
        for v in child:
            if v.tag == 'vert':
                verts.append(numpy.array(map(float,v.text.split())))
            elif v.tag == 'diameter':
                diameter = float(v.text)
            else:
                raise ValueError('unknown tag: %s'%v.tag)
        return {'verts':verts, 'diameter':diameter}

    def get_tvtk_actors(self):
        actors = []
        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                info = self._get_info_for_cylindrical_arena(child)
                actors.extend( experiment_layout.cylindrical_arena(info=info))
            elif child.tag == 'cylindrical_post':
                info = self._get_info_for_cylindrical_post(child)
                actors.extend( experiment_layout.cylindrical_post(info=info))
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)
        return actors

    def plot_stim_over_distorted_image( self, ax, cam_id ):
       # we want to work with scaled coordinates
        R = self._get_reconstructor()
        R = R.get_scaled()

        class Projection:
            def __init__(self,R,cam_id):
                self.R = R
                self.cam_id = cam_id
            def project(self,X):
                return self.R.find2d(self.cam_id,X,distorted=True)
        P = Projection(R,cam_id)
        self.plot_stim( ax, projection=P.project )

    def plot_stim( self, ax, projection=None ):
        assert projection is not None

        plotted_anything = False

        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                plotted_anything = True
                info = self._get_info_for_cylindrical_arena(child)
                assert numpy.allclose(info['axis'],numpy.array([0,0,1])), "only vertical areas supported at the moment"

                N = 128
                theta = numpy.linspace(0,2*numpy.pi,N)
                r = info['diameter']/2.0
                xs = r*numpy.cos( theta ) + info['origin'][0]
                ys = r*numpy.sin( theta ) + info['origin'][1]

                z_levels = numpy.linspace(info['origin'][2],info['origin'][2]+info['height'],5)
                for z in z_levels:
                    plotx, ploty = [],[]
                    for (x,y) in zip(xs,ys):
                        X = x,y,z
                        x2d, y2d = projection(X)
                        plotx.append(x2d); ploty.append(y2d)
                    ax.plot( plotx, ploty, 'k-' )
            elif child.tag == 'cylindrical_post':
                plotted_anything = True
                info = self._get_info_for_cylindrical_post(child)
                xs, ys = [], []
                # XXX TODO: extrude line into cylinder
                for X in info['verts']:
                    v2 = projection(X)
                    xs.append(v2[0]); ys.append(v2[1])
                ax.plot( xs, ys, 'k-', linewidth=5 )
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)
        if not plotted_anything:
            import warnings
            warnings.warn("Did not plot any stimulus")

class StimulusFanout(object):
    def __init__(self,root):
        assert root.tag == 'stimulus_fanout_xml'
        assert root.attrib['version']=='1'
        self.root = root
    def _get_episode_for_timestamp( self, timestamp_string=None ):
        for single_episode in self.root.findall("single_episode"):
            for kh5_file in single_episode.findall("kh5_file"):
                fname = kh5_file.attrib['name']
                fname_timestamp_string = os.path.splitext(fname)[0][4:]
                if fname_timestamp_string == timestamp_string:
                    if 1:
                        # check that the file has not changed
                        #print 'fname',fname
                        expected_md5 = kh5_file.attrib['md5sum']
                        m = hashlib.md5()
                        m.update(open(fname,mode='rb').read())
                        actual_md5 = m.hexdigest()
                        #print expected_md5
                        #print actual_md5
                        assert expected_md5==actual_md5
                    stim_fname = single_episode.find("stimxml_file").attrib['name']
                    return single_episode, kh5_file, stim_fname
        raise ValueError("could not find timestamp_string '%s'"%timestamp_string)
    def get_walking_start_stops_for_timestamp( self, timestamp_string=None ):
        single_episode, kh5_file, stim_fname = self._get_episode_for_timestamp(timestamp_string=timestamp_string)
        start_stops = []
        for walking in single_episode.findall("walking"):
            start = walking.attrib.get('start',None) # frame number
            stop = walking.attrib.get('stop',None) # frame number
            if start is not None: start = int(start)
            if stop is not None: stop = int(stop)
            start_stops.append( (start,stop) )
        return start_stops
    def get_obj_ids_for_timestamp( self, timestamp_string=None ):
        single_episode, kh5_file, stim_fname = self._get_episode_for_timestamp(timestamp_string=timestamp_string)
        include_ids = None
        exclude_ids = None
        for include in kh5_file.findall('include'):
            if include_ids is None:
                include_ids = []
            if include.text is not None:
                obj_ids = parse_seq( include.text )
                include_ids.extend( obj_ids )
            else:
                obj_ids = None
        for exclude in kh5_file.findall('exclude'):
            if exclude_ids is None:
                exclude_ids = []
            if exclude.text is not None:
                obj_ids = parse_seq( exclude.text )
                exclude_ids.extend( obj_ids )
            else:
                obj_ids = None
        return include_ids, exclude_ids
    def get_stimulus_for_timestamp( self, timestamp_string=None ):
        single_episode, kh5_file, stim_fname = self._get_episode_for_timestamp(timestamp_string=timestamp_string)
        root = ET.parse(stim_fname).getroot()
        stim = Stimulus(root)
        #stim.verify_timestamp( fname_timestamp_string )
        return stim

def xml_fanout_from_filename( filename ):
    root = ET.parse(filename).getroot()
    sf = StimulusFanout(root)
    return sf

def xml_stimulus_from_filename( filename, timestamp_string=None ):
    root = ET.parse(filename).getroot()
    if root.tag == 'stimxml':
        return Stimulus(root)
    elif root.tag == 'stimulus_fanout_xml':
        assert timestamp_string is not None
        sf = xml_fanout_from_filename( filename )
        stim = sf.get_stimulus_for_timestamp( timestamp_string=timestamp_string )
        return stim
    else:
        raise ValueError('unknown XML file')

def print_kh5_files_in_fanout(filename):
    sf = xml_fanout_from_filename( filename )
    for single_episode in sf.root.findall("single_episode"):
        for kh5_file in single_episode.findall('kh5_file'):
            print kh5_file.attrib['name'],
    print

def main():
    import sys
    filename = sys.argv[1]
    print_kh5_files_in_fanout( filename )

if __name__=='__main__':
    main()
