import xml.etree.ElementTree as ET
import flydra.reconstruct as reconstruct
import flydra.a2.experiment_layout as experiment_layout

class Stimulus(object):

    def __init__(self,root):
        assert root.tag == 'stimxml'
        assert root.attrib['version']=='1'
        self.root = root

    def verify_reconstructor(self,other_R):
        r_node = self.root.find("multi_camera_reconstructor")
        R = reconstruct.Reconstructor_from_xml(r_node)
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

    def get_tvtk_actors(self):
        actors = []
        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
                pass
            elif child.tag == 'cylindrical_arena':
                import warnings
                warnings.warn("Not drawing arena")
            elif child.tag == 'cylindrical_post':
                verts = []
                for v in child:
                    if v.tag == 'vert':
                        verts.append(map(float,v.text.split()))
                    if v.tag == 'diameter':
                        diameter = float(v.text)
                actors = experiment_layout.cylindrical_post(verts=verts,
                                                            diameter=diameter)
            else:
                import warnings
                warnings.warn("Unknown node: %s"%child.tag)
        return actors

def xml_stimulus_from_filename( filename ):
    root = ET.parse(filename).getroot()
    return Stimulus(root)
