#!/usr/bin/env python

# Starting point: mayavi2-2.0.2a1/enthought.mayavi/examples/standalone.py
from __future__ import print_function
from os.path import join, dirname
import warnings
from tvtk.api import tvtk
from pyface.api import GUI

# The core Engine.
from mayavi.core.engine import Engine
from mayavi.core.ui.engine_view import EngineView

# Usual MayaVi imports
from mayavi.sources.api import VTKXMLFileReader
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.modules.api import Outline, ScalarCutPlane, Streamline
import numpy
import numpy as np
import tables
import sys
import json

import mayavi.tools.sources as sources
from mayavi.sources.array_source import ArraySource
from mayavi.modules.vectors import Vectors

import traits.api as traits
from traitsui.api import View, Item, Group, Handler, HGroup, VGroup, RangeEditor

import argparse

import flydra_analysis.talign as talign
import flydra_core.reconstruct as reconstruct
import flydra_analysis.a2.core_analysis as core_analysis
import flydra_analysis.a2.xml_stimulus as xml_stimulus
import flydra_analysis.analysis.result_utils as result_utils

import cgtypes  # import cgkit 1.x

from pyface.api import Widget, Window
from tvtk.pyface.api import Scene, DecoratedScene
from pyface.api import SplitApplicationWindow
from pyface.api import FileDialog, OK


def hom2vtk(arr):
    """convert 3D homogeneous coords to VTK"""
    return (arr[:3, :] / arr[3, :]).T


class CalibrationAlignmentWindow(Widget):
    params = traits.Instance(talign.Alignment)
    save_align_json = traits.Button(label="Save alignment data as .json file")
    save_new_cal = traits.Button(label="Save new calibration as .xml file")
    save_new_cal_dir = traits.Button(label="Save new calibration as directory")

    traits_view = View(
        Group(
            (
                Item("params", style="custom", show_label=False),
                Item("save_align_json", show_label=False),
                Item("save_new_cal", show_label=False),
                Item("save_new_cal_dir", show_label=False),
            )
        ),
        title="Calibration Alignment",
    )
    orig_data_verts = traits.Instance(object)
    orig_data_speeds = traits.Instance(object)
    reconstructor = traits.Instance(object)
    viewed_data = traits.Instance(tvtk.DataSet)
    source = traits.Instance(VTKDataSource)

    def __init__(self, parent, **traits):
        super(CalibrationAlignmentWindow, self).__init__(**traits)
        self.params = talign.Alignment()

        self.control = self.edit_traits(
            parent=parent,
            kind="subpanel",
            context={"h1": self.params, "object": self},  # XXX ???
        ).control
        self.params.on_trait_change(self._params_changed)

    def set_data(self, orig_data_verts, orig_data_speeds, reconstructor, align_json):
        self.orig_data_verts = orig_data_verts
        self.orig_data_speeds = orig_data_speeds
        self.reconstructor = reconstructor

        assert orig_data_verts.ndim == 2
        assert orig_data_speeds.ndim == 1
        assert orig_data_verts.shape[0] == 4
        assert orig_data_verts.shape[1] == orig_data_speeds.shape[0]

        # from mayavi2-2.0.2a1/enthought.tvtk/enthought/tvtk/tools/mlab.py
        #   Glyphs.__init__
        points = hom2vtk(orig_data_verts)
        polys = numpy.arange(0, len(points), 1, "l")
        polys = numpy.reshape(polys, (len(points), 1))
        pd = tvtk.PolyData(points=points, polys=polys)
        pd.point_data.scalars = orig_data_speeds
        pd.point_data.scalars.name = "speed"
        self.viewed_data = pd
        self.source = VTKDataSource(data=self.viewed_data, name="aligned data")

        if align_json:
            j = json.loads(open(align_json).read())
            self.params.s = j["s"]
            for i, k in enumerate(("tx", "ty", "tz")):
                setattr(self.params, k, j["t"][i])

            R = np.array(j["R"])
            rx, ry, rz = np.rad2deg(euler_from_matrix(R, "sxyz"))

            self.params.r_x = rx
            self.params.r_y = ry
            self.params.r_z = rz

            self._params_changed()

    def _params_changed(self):
        if self.orig_data_verts is None or self.viewed_data is None:
            # no data set yet
            return
        M = self.params.get_matrix()
        verts = np.dot(M, self.orig_data_verts)
        self.viewed_data.points = hom2vtk(verts)
        self.source.update()
        self.source.render()

    def get_aligned_R(self):
        M = self.params.get_matrix()
        R = self.reconstructor
        alignedR = R.get_aligned_copy(M, update_water_boundary=False)
        return alignedR

    def _save_align_json_fired(self):
        wildcard = "JSON files (*.json)|*.json|" + FileDialog.WILDCARD_ALL
        dialog = FileDialog(  # parent=self.window.control,
            title="Save alignment as .json file", action="save as", wildcard=wildcard
        )
        if dialog.open() == OK:
            buf = json.dumps(self.params.as_dict())
            with open(dialog.path, mode="w") as fd:
                fd.write(buf)

    def _save_new_cal_fired(self):
        wildcard = "XML files (*.xml)|*.xml|" + FileDialog.WILDCARD_ALL
        dialog = FileDialog(  # parent=self.window.control,
            title="Save calibration .xml file", action="save as", wildcard=wildcard
        )
        if dialog.open() == OK:
            alignedR = self.get_aligned_R()
            alignedR.save_to_xml_filename(dialog.path)

    def _save_new_cal_dir_fired(self):
        dialog = FileDialog(  # parent=self.window.control,
            title="Save calibration directory", action="save as",
        )
        if dialog.open() == OK:
            alignedR = self.get_aligned_R()
            alignedR.save_to_files_in_new_directory(dialog.path)


class IVTKWithCalGUI(SplitApplicationWindow):
    # The ratio of the size of the left/top pane to the right/bottom pane.
    ratio = traits.Float(0.7)

    # The direction in which the panel is split.
    direction = traits.Str("vertical")

    # The `Scene` instance into which VTK renders.
    scene = traits.Instance(Scene)

    cal_align = traits.Instance(CalibrationAlignmentWindow)

    ###########################################################################
    # 'object' interface.
    ###########################################################################
    def __init__(self, **traits):
        """ Creates a new window. """

        # Base class constructor.
        super(IVTKWithCalGUI, self).__init__(**traits)
        self.title = "Calibration Alignment GUI"
        # Create the window's menu bar.
        # self.menu_bar_manager = create_ivtk_menu(self)

    ###########################################################################
    # Protected 'SplitApplicationWindow' interface.
    ###########################################################################
    def _create_lhs(self, parent):
        """ Creates the left hand side or top depending on the style. """
        self.scene = DecoratedScene(parent)
        return self.scene.control

    def _create_rhs(self, parent):
        """ Creates the right hand side or bottom depending on the
        style.  's' and 'scene' are bound to the Scene instance."""

        self.cal_align = CalibrationAlignmentWindow(parent)
        return self.cal_align.control


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filename", nargs="+", help="name of flydra .hdf5 file",
    )

    parser.add_argument(
        "--stim-xml",
        type=str,
        default=None,
        help="name of XML file with stimulus info",
        required=True,
    )

    parser.add_argument(
        "--align-json",
        type=str,
        default=None,
        help="previously exported json file containing s,R,T",
    )

    parser.add_argument(
        "--radius",
        type=float,
        help="radius of line (in meters)",
        default=0.002,
        metavar="RADIUS",
    )

    parser.add_argument("--obj-only", type=str)

    parser.add_argument(
        "--obj-filelist", type=str, help="use object ids from list in text file",
    )

    parser.add_argument(
        "-r",
        "--reconstructor",
        dest="reconstructor_path",
        type=str,
        help=("calibration/reconstructor path (if not specified, " "defaults to FILE)"),
    )

    args = parser.parse_args()
    options = args  # optparse OptionParser backwards compatibility

    reconstructor_path = args.reconstructor_path
    fps = None

    ca = core_analysis.get_global_CachingAnalyzer()
    by_file = {}

    for h5_filename in args.filename:
        assert tables.is_hdf5_file(h5_filename)
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(
            h5_filename
        )
        this_fps = result_utils.get_fps(data_file, fail_on_error=False)
        if fps is None:
            if this_fps is not None:
                fps = this_fps
        if reconstructor_path is None:
            reconstructor_path = data_file
        by_file[h5_filename] = (use_obj_ids, data_file)
    del h5_filename
    del obj_ids, use_obj_ids, is_mat_file, data_file, extra

    if options.obj_only is not None:
        obj_only = core_analysis.parse_seq(options.obj_only)
    else:
        obj_only = None

    if reconstructor_path is None:
        raise RuntimeError("must specify reconstructor from CLI if not using .h5 files")

    R = reconstruct.Reconstructor(reconstructor_path)

    if fps is None:
        fps = 100.0
        warnings.warn("Setting fps to default value of %f" % fps)
    else:
        fps = 1.0

    if options.stim_xml is None:
        raise ValueError(
            "stim_xml must be specified (how else will you align the data?"
        )

    if 1:
        stim_xml = xml_stimulus.xml_stimulus_from_filename(options.stim_xml,)
        try:
            fanout = xml_stimulus.xml_fanout_from_filename(options.stim_xml)
        except xml_stimulus.WrongXMLTypeError:
            pass
        else:
            include_obj_ids, exclude_obj_ids = fanout.get_obj_ids_for_timestamp(
                timestamp_string=file_timestamp
            )
            if include_obj_ids is not None:
                use_obj_ids = include_obj_ids
            if exclude_obj_ids is not None:
                use_obj_ids = list(set(use_obj_ids).difference(exclude_obj_ids))
            print("using object ids specified in fanout .xml file")
        if stim_xml.has_reconstructor():
            stim_xml.verify_reconstructor(R)

    x = []
    y = []
    z = []
    speed = []

    if options.obj_filelist is not None:
        obj_filelist = options.obj_filelist
    else:
        obj_filelist = None

    if obj_filelist is not None:
        obj_only = 1

    if obj_only is not None:
        if len(by_file) != 1:
            raise RuntimeError("specifying obj_only can only be done for a single file")
        if obj_filelist is not None:
            data = np.loadtxt(obj_filelist, delimiter=",")
            obj_only = np.array(data[:, 0], dtype="int")
            print(obj_only)

        use_obj_ids = numpy.array(obj_only)
        h5_filename = by_file.keys()[0]
        (prev_use_ob_ids, data_file) = by_file[h5_filename]
        by_file[h5_filename] = (use_obj_ids, data_file)

    for h5_filename in by_file:
        (use_obj_ids, data_file) = by_file[h5_filename]
        for obj_id_enum, obj_id in enumerate(use_obj_ids):
            rows = ca.load_data(
                obj_id,
                data_file,
                use_kalman_smoothing=False,
                # dynamic_model_name = dynamic_model_name,
                # frames_per_second=fps,
                # up_dir=up_dir,
            )
            verts = numpy.array([rows["x"], rows["y"], rows["z"]]).T
            if len(verts) >= 3:
                verts_central_diff = verts[2:, :] - verts[:-2, :]
                dt = 1.0 / fps
                vels = verts_central_diff / (2 * dt)
                speeds = numpy.sqrt(numpy.sum(vels ** 2, axis=1))
                # pad end points
                speeds = numpy.array([speeds[0]] + list(speeds) + [speeds[-1]])
            else:
                speeds = numpy.zeros((verts.shape[0],))

            if verts.shape[0] != len(speeds):
                raise ValueError("mismatch length of x data and speeds")
            x.append(verts[:, 0])
            y.append(verts[:, 1])
            z.append(verts[:, 2])
            speed.append(speeds)
        data_file.close()
    del h5_filename, use_obj_ids, data_file

    if 0:
        # debug
        if stim_xml is not None:
            v = None
            for child in stim_xml.root:
                if child.tag == "cubic_arena":
                    info = stim_xml._get_info_for_cubic_arena(child)
                    v = info["verts4x4"]
            if v is not None:
                for vi in v:
                    print("adding", vi)
                    x.append([vi[0]])
                    y.append([vi[1]])
                    z.append([vi[2]])
                    speed.append([100.0])

    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    w = np.ones_like(x)
    speed = np.concatenate(speed)

    # homogeneous coords
    verts = np.array([x, y, z, w])

    #######################################################

    # Create the MayaVi engine and start it.
    e = Engine()
    # start does nothing much but useful if someone is listening to
    # your engine.
    e.start()

    # Create a new scene.
    from tvtk.tools import ivtk

    # viewer = ivtk.IVTK(size=(600,600))
    viewer = IVTKWithCalGUI(size=(800, 600))
    viewer.open()
    e.new_scene(viewer)

    viewer.cal_align.set_data(verts, speed, R, args.align_json)

    if 0:
        # Do this if you need to see the MayaVi tree view UI.
        ev = EngineView(engine=e)
        ui = ev.edit_traits()

    # view aligned data
    e.add_source(viewer.cal_align.source)

    v = Vectors()
    v.glyph.scale_mode = "data_scaling_off"
    v.glyph.color_mode = "color_by_scalar"
    v.glyph.glyph_source.glyph_position = "center"
    v.glyph.glyph_source.glyph_source = tvtk.SphereSource(radius=options.radius,)
    e.add_module(v)

    if stim_xml is not None:
        if 0:
            stim_xml.draw_in_mayavi_scene(e)
        else:
            actors = stim_xml.get_tvtk_actors()
            viewer.scene.add_actors(actors)

    gui = GUI()
    gui.start_event_loop()


# --------------------------------------------------------------------------
# Below here is copied from so that we do not need to import ROS.
# https://github.com/davheld/tf/blob/master/src/tf/transformations.py

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


if __name__ == "__main__":
    main()
