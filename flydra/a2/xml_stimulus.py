import xml.etree.ElementTree as ET
import flydra.reconstruct as reconstruct
import flydra.a2.experiment_layout as experiment_layout

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


    def _get_info_for_cylindrical_post(self,child):
        verts = []
        for v in child:
            if v.tag == 'vert':
                verts.append(map(float,v.text.split()))
            if v.tag == 'diameter':
                diameter = float(v.text)
        return {'verts':verts, 'diameter':diameter}

    def get_tvtk_actors(self):
        actors = []
        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                import warnings
                warnings.warn("Not drawing arena")
            elif child.tag == 'cylindrical_post':
                info = self._get_info_for_cylindrical_post(child)
                actors = experiment_layout.cylindrical_post(verts=info['verts'],
                                                            diameter=info['diameter'])
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)
        return actors

    def plot_stim( self, ax, cam_id ):
        # we want to work with scaled coordinates
        R = self._get_reconstructor()
        R = R.get_scaled()

        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                import warnings
                warnings.warn("Not drawing arena")
            elif child.tag == 'cylindrical_post':
                info = self._get_info_for_cylindrical_post(child)
                xs, ys = [], []
                # XXX TODO: extrude line into cylinder
                for v in info['verts']:
                    v2 = R.find2d(cam_id,v,distorted=True)
                    xs.append(v2[0]); ys.append(v2[1])
                ax.plot( xs, ys, 'ko-' )
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)


def xml_stimulus_from_filename( filename ):
    root = ET.parse(filename).getroot()
    return Stimulus(root)
