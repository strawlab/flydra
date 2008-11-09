#!/usr/bin/env python

# Starting point: mayavi2-2.0.2a1/enthought.mayavi/examples/standalone.py

from os.path import join, dirname
from enthought.tvtk.api import tvtk
from enthought.pyface.api import GUI
# The core Engine.
from enthought.mayavi.engine import Engine
from enthought.mayavi.view.engine_view import EngineView
# Usual MayaVi imports
from enthought.mayavi.sources.api import VTKXMLFileReader, VTKDataSource
from enthought.mayavi.modules.api import Outline, ScalarCutPlane, Streamline
import numpy
import numpy as np

import enthought.mayavi.tools.sources as sources
from enthought.mayavi.sources.array_source import ArraySource
from enthought.mayavi.modules.surface import Surface
from enthought.mayavi.modules.vectors import Vectors

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group, Handler, HGroup, \
     VGroup, RangeEditor

from optparse import OptionParser

import flydra.align
import flydra.reconstruct as reconstruct
import flydra.a2.core_analysis as core_analysis
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.analysis.result_utils as result_utils

import cgtypes # import cgkit 1.x

D2R = np.pi/180.0

def hom2vtk(arr):
    """convert 3D homogeneous coords to VTK"""
    return (arr[:3,:]/arr[3,:]).T

def cgmat2np(cgkit_mat):
    """convert cgkit matrix to numpy matrix"""
    arr = np.array(cgkit_mat.toList())
    if len(arr)==9:
        arr.shape = 3,3
    elif len(arr)==16:
        arr.shape = 4,4
    else:
        raise ValueError('unknown shape')
    return arr.T

def test_cgmat2mp():
    point1 = (1,0,0)
    point1_out = (0,1,0)

    cg_quat = cgtypes.quat().fromAngleAxis( 90.0*D2R, (0,0,1))
    cg_in = cgtypes.vec3(point1)

    m_cg = cg_quat.toMat3()
    cg_out = m_cg*cg_in
    cg_out_tup = (cg_out[0],cg_out[1],cg_out[2])
    assert np.allclose( cg_out_tup, point1_out)

    m_np = cgmat2np(m_cg)
    np_out = np.dot( m_np, point1 )
    assert np.allclose( np_out, point1_out)

class CalibrationAlignment(traits.HasTraits):
    s = traits.Float(1.0)
    tx = traits.Float(0)
    ty = traits.Float(0)
    tz = traits.Float(0)

    r_x = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)
    r_y = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)
    r_z = traits.Range(-180.0,180.0, 0.0,mode='slider',set_enter=True)

    new_cal_filename = traits.File(extension='.xml')
    save_new_cal = traits.Button(label='Save new calibration as .xml file')
    save_new_cal_dir = traits.Button(label='Save new calibration as directory')

    traits_view = View( Group( ( Item('s'),
                                 Item('tx'),
                                 Item('ty'),
                                 Item('tz'),
                                 Item('r_x',style='custom'),
                                 Item('r_y',style='custom'),
                                 Item('r_z',style='custom'),
                                 Item('new_cal_filename'),
                                 Item( 'save_new_cal', show_label = False ),
                                 Item( 'save_new_cal_dir', show_label = False ),
                                 )),
                        title = 'Calibration Alignment Parameters',
                        )

    def __init__(self,orig_data_verts,orig_data_speeds,reconstructor):
        self.orig_data_verts = orig_data_verts
        self.orig_data_speeds = orig_data_speeds
        self.reconstructor = reconstructor

        assert orig_data_verts.ndim == 2
        assert orig_data_speeds.ndim == 1
        assert orig_data_verts.shape[0] == 4
        assert orig_data_verts.shape[1] == orig_data_speeds.shape[0]


        # from mayavi2-2.0.2a1/enthought.tvtk/enthought/tvtk/tools/mlab.py Glyphs.__init__:
        points = hom2vtk(orig_data_verts)
        polys = numpy.arange(0, len(points), 1, 'l')
        polys = numpy.reshape(polys, (len(points), 1))
        pd = tvtk.PolyData(points=points, polys=polys)
        pd.point_data.scalars = orig_data_speeds
        pd.point_data.scalars.name = 'speed'
        self.viewed_data = VTKDataSource(data=pd)

    def get_sRt(self):
        qx = cgtypes.quat().fromAngleAxis( self.r_x*D2R, cgtypes.vec3(1,0,0))
        qy = cgtypes.quat().fromAngleAxis( self.r_y*D2R, cgtypes.vec3(0,1,0))
        qz = cgtypes.quat().fromAngleAxis( self.r_z*D2R, cgtypes.vec3(0,0,1))
        Rx = cgmat2np(qx.toMat3())
        Ry = cgmat2np(qy.toMat3())
        Rz = cgmat2np(qz.toMat3())
        R = np.dot(Rx, np.dot(Ry,Rz))

        t = np.array([self.tx, self.ty, self.tz],np.float)

        return self.s, R, t

    def _anytrait_changed(self):
        s,R,t = self.get_sRt()
        M = flydra.align.build_xform( s, R, t)

        verts = np.dot(M,self.orig_data_verts)

        self.viewed_data.data.points = hom2vtk(verts)

        self.viewed_data.data.modified()
        self.viewed_data.update()

    def get_aligned_scaled_R(self):
        s,R,t = self.get_sRt()
        scaled = self.reconstructor
        alignedR = scaled.get_aligned_copy(s,R,t)
        return alignedR

    def _save_new_cal_fired(self):
        alignedR = self.get_aligned_scaled_R()
        alignedR.save_to_xml_filename(self.new_cal_filename)
        print 'saved calibration to XML file...',self.new_cal_filename

    def _save_new_cal_dir_fired(self):
        alignedR = self.get_aligned_scaled_R()
        alignedR.save_to_files_in_new_directory(self.new_cal_filename)
        print 'saved calibration to directory',self.new_cal_filename

def main():
    usage = '%prog FILE [options]'

    # A man page can be generated with:
    # 'help2man -N -n kdviewer kdviewer > kdviewer.1'

    parser = OptionParser(usage)

    parser.add_option("--stim-xml",
                      type="string",
                      default=None,
                      help="name of XML file with stimulus info",
                      )

    parser.add_option("--radius", type="float",
                      help="radius of line (in meters)",
                      default=0.002,
                      metavar="RADIUS")

    parser.add_option("--obj-only", type="string")

    (options, args) = parser.parse_args()
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    h5_filename=args[0]

    if options.obj_only is not None:
        obj_only = core_analysis.parse_seq(options.obj_only)
    else:
        obj_only = None

    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(h5_filename)
    R = reconstruct.Reconstructor(data_file).get_scaled()
    fps = result_utils.get_fps( data_file, fail_on_error=False )

    if fps is None:
        fps = 100.0
        warnings.warn('Setting fps to default value of %f'%fps)

    if options.stim_xml is not None:
        file_timestamp = data_file.filename[4:19]
        stim_xml = xml_stimulus.xml_stimulus_from_filename(options.stim_xml,
                                                           timestamp_string=file_timestamp,
                                                           )
        try:
            fanout = xml_stimulus.xml_fanout_from_filename( options.stim_xml )
        except xml_stimulus.WrongXMLTypeError:
            pass
        else:
            include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp( timestamp_string=file_timestamp )
            if include_obj_ids is not None:
                use_obj_ids = include_obj_ids
            if exclude_obj_ids is not None:
                use_obj_ids = list( set(use_obj_ids).difference( exclude_obj_ids ) )
            print 'using object ids specified in fanout .xml file'
        if stim_xml.has_reconstructor():
            stim_xml.verify_reconstructor(R)
    else:
        stim_xml = None

    x = []
    y = []
    z = []
    speed = []
    if obj_only is not None:
        use_obj_ids = numpy.array(obj_only)

    for obj_id_enum,obj_id in enumerate(use_obj_ids):
        rows = ca.load_data( obj_id, data_file,
                                use_kalman_smoothing=False,
                                #dynamic_model_name = dynamic_model_name,
                                #frames_per_second=fps,
                                #up_dir=up_dir,
                                )
        verts = numpy.array( [rows['x'], rows['y'], rows['z']] ).T
        if len(verts)>=3:
            verts_central_diff = verts[2:,:] - verts[:-2,:]
            dt = 1.0/fps
            vels = verts_central_diff/(2*dt)
            speeds = numpy.sqrt(numpy.sum(vels**2,axis=1))
            speeds = numpy.array([speeds[0]] + list(speeds) + [speeds[-1]]) # pad end points
        else:
            speeds = numpy.zeros( (verts.shape[1],) )

        x.append( rows['x'] )
        y.append( rows['y'] )
        z.append( rows['z'] )
        speed.append(speeds)
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    w = np.ones_like(x)
    speed = np.concatenate(speed)

    # homogeneous coords
    verts = np.array([x,y,z,w])

    #######################################################

    # Create the MayaVi engine and start it.
    e = Engine()
    # start does nothing much but useful if someone is listening to
    # your engine.
    e.start()

    cal_align = CalibrationAlignment(verts,speed,R)

    cal_align.edit_traits()

    # Create a new scene.
    scene = e.new_scene()
    if stim_xml is not None:
        actors = stim_xml.get_tvtk_actors()
        scene.scene.add_actors(actors)

    if 0:
        # Do this if you need to see the MayaVi tree view UI.
        ev = EngineView(engine=e)
        ui = ev.edit_traits()

    e.add_source(cal_align.viewed_data)

    v = Vectors()
    v.glyph.scale_mode = 'data_scaling_off'
    v.glyph.color_mode = 'color_by_scalar'
    v.glyph.glyph_source = tvtk.SphereSource(radius=options.radius)
    e.add_module(v)

    gui = GUI()
    gui.start_event_loop()

if __name__ == '__main__':
    main()
