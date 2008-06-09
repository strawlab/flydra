import xml.etree.ElementTree as ET
import flydra.reconstruct as reconstruct
import flydra.a2.experiment_layout as experiment_layout
import numpy

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

        for child in self.root:
            if child.tag in ['multi_camera_reconstructor','valid_h5_times']:
                continue
            elif child.tag == 'cylindrical_arena':
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

def xml_stimulus_from_filename( filename ):
    root = ET.parse(filename).getroot()
    return Stimulus(root)
