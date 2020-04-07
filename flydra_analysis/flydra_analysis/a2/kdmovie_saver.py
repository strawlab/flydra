from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import os, sys, math

sys.path.insert(0, os.curdir)
import tvtk
import numpy
import numpy as np
import tables as PT
from optparse import OptionParser
from . import core_analysis
from . import stimulus_positions
import scipy.io
from . import conditions
import flydra_analysis.a2.xml_stimulus as xml_stimulus
import flydra_core.reconstruct as reconstruct
import pkg_resources
from flydra_analysis.a2.pos_ori2fu import pos_ori2fu
import cgtypes  # cgtypes 1.x


class AnimationPath(object):
    def __init__(self, fname):
        fd = open(fname, mode="r")
        data = []
        for line in fd.readlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if not len(line):
                continue
            split_line = line.split()
            fl = map(float, split_line)
            data.append(fl)
        self.data = numpy.array(data)
        print("self.data", self.data)

    def get_pos_ori(self, t):
        file_ts = self.data[:, 0]
        tdiff = file_ts[1:] - file_ts[:-1]
        if tdiff.min() < 0.0:
            raise ValueError("animation path times go backwards!")
        t = t % file_ts[-1]  # wrap around
        lower_idx = numpy.nonzero((file_ts <= t))[0][-1]
        upper_idx = lower_idx + 1
        lower_t = file_ts[lower_idx]
        file_dt = file_ts[upper_idx] - lower_t
        frac = (t - lower_t) / file_dt

        pos_lower = self.data[lower_idx, 1:4]
        ori_lower = cgtypes.quat(self.data[lower_idx, 4:8])

        pos_upper = self.data[upper_idx, 1:4]
        ori_upper = cgtypes.quat(self.data[upper_idx, 4:8])

        pos = frac * (pos_upper - pos_lower) + pos_lower
        if ori_lower != ori_upper:
            ori = cgtypes.slerp(frac, ori_lower, ori_upper)
        else:
            ori = ori_lower

        if np.isnan(ori.w):
            raise ValueError("orientation is nan")
        return pos, ori


def doit(
    filename,
    obj_only=None,
    min_length=10,
    use_kalman_smoothing=True,
    data_fps=100.0,
    save_fps=25,
    vertical_scale=False,
    max_vel="auto",
    draw_stim_func_str=None,
    floor=True,
    animation_path_fname=None,
    output_dir=".",
    cam_only_move_duration=5.0,
    options=None,
):

    if not use_kalman_smoothing:
        if dynamic_model_name is not None:
            print(
                "ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model option (--dynamic-model)",
                file=sys.stderr,
            )
            sys.exit(1)
    dynamic_model_name = options.dynamic_model

    if animation_path_fname is None:
        animation_path_fname = pkg_resources.resource_filename(
            __name__, "kdmovie_saver_default_path.kmp"
        )
    camera_animation_path = AnimationPath(animation_path_fname)

    mat_data = None
    try:
        try:
            data_path, data_filename = os.path.split(filename)
            data_path = os.path.expanduser(data_path)
            sys.path.insert(0, data_path)
            mat_data = scipy.io.mio.loadmat(data_filename)
        finally:
            del sys.path[0]
    except IOError as err:
        print(
            "not a .mat file at %s, treating as .hdf5 file"
            % (os.path.join(data_path, data_filename))
        )

    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(filename)

    if obj_only is not None:
        use_obj_ids = obj_only

    if options.stim_xml is not None:
        file_timestamp = data_file.filename[4:19]
        stim_xml = xml_stimulus.xml_stimulus_from_filename(
            options.stim_xml, timestamp_string=file_timestamp,
        )
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

    if dynamic_model_name is None:
        if "dynamic_model_name" in extra:
            dynamic_model_name = extra["dynamic_model_name"]
            print('detected file loaded with dynamic model "%s"' % dynamic_model_name)
            if dynamic_model_name.startswith("EKF "):
                dynamic_model_name = dynamic_model_name[4:]
            print('  for smoothing, will use dynamic model "%s"' % dynamic_model_name)
        else:
            print(
                "no dynamic model name specified, and it could not be determined from the file, either"
            )

    filename_trimmed = os.path.split(os.path.splitext(filename)[0])[-1]

    assert use_obj_ids is not None

    #################
    rw = tvtk.RenderWindow(size=(1024, 768))

    ren = tvtk.Renderer(background=(1.0, 1.0, 1.0))
    camera = ren.active_camera

    rw.add_renderer(ren)

    lut = tvtk.LookupTable(hue_range=(0.667, 0.0))
    #################
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if len(use_obj_ids) == 1:
        animate_path = True
        # allow path to grow during trajectory
    else:
        animate_path = False
        obj_verts = []
        speeds = []

    for obj_id in use_obj_ids:

        print("loading %d" % obj_id)
        results = ca.calculate_trajectory_metrics(
            obj_id,
            data_file,
            use_kalman_smoothing=use_kalman_smoothing,
            frames_per_second=data_fps,
            dynamic_model_name=dynamic_model_name,
            # method='position based',
            method_params={"downsample": 1,},
        )

        if len(use_obj_ids) == 1:
            obj_verts = results["X_kalmanized"]
            speeds = results["speed_kalmanized"]
            real_frames = results["frame"]

        else:
            obj_verts.append(results["X_kalmanized"])
            speeds.append(results["speed_kalmanized"])
            real_frames.append(results["frame"])

    if options.start is not None:
        good_cond = real_frames >= options.start
        obj_verts = obj_verts[good_cond]
        speeds = speeds[good_cond]
        real_frames = real_frames[good_cond]

    if not len(use_obj_ids) == 1:
        obj_verts = numpy.concatenate(obj_verts, axis=0)
        speeds = numpy.concatenate(speeds, axis=0)
        real_frames = numpy.concatenate(real_frames, axis=0)

    ####################### start draw permanently on stuff ############################

    if options.stim_xml is not None:

        if not is_mat_file:
            R = reconstruct.Reconstructor(data_file)
            stim_xml.verify_reconstructor(R)

        if not is_mat_file:
            assert data_file.filename.startswith("DATA") and (
                data_file.filename.endswith(".h5")
                or data_file.filename.endswith(".kh5")
            )
            file_timestamp = data_file.filename[4:19]
        actors = stim_xml.get_tvtk_actors()
        for actor in actors:
            ren.add_actor(actor)

    if 1:
        if 0:
            # Inspired by pyface.tvtk.decorated_scene
            marker = tvtk.OrientationMarkerWidget()

        axes = tvtk.AxesActor()
        axes.set(
            # normalized_tip_length=(0.04, 0.4, 0.4),
            # normalized_shaft_length=(0.6, 0.6, 0.6),
            shaft_type="cylinder",
            total_length=(0.15, 0.15, 0.15),
        )

        if 1:
            axes.x_axis_label_text = ""
            axes.y_axis_label_text = ""
            axes.z_axis_label_text = ""
        else:
            p = axes.x_axis_caption_actor2d.caption_text_property
            axes.y_axis_caption_actor2d.caption_text_property = p
            axes.z_axis_caption_actor2d.caption_text_property = p
            p.color = 0.0, 0.0, 0.0  # black
        # axes.camera = camera
        # axes.attachment_point_coordinate = (0,0,0)
        axes.origin = (-0.5, 1, 0)

        if 0:
            rwi = rw.interactor
            print("rwi", rwi)
            marker.orientation_marker = axes
            # marker.interactive = False
            marker.interactor = rwi
            marker.enabled = True

        ren.add_actor(axes)

    #######################

    if max_vel == "auto":
        max_vel = speeds.max()
    else:
        max_vel = float(max_vel)
    vel_mapper = tvtk.PolyDataMapper()
    vel_mapper.lookup_table = lut
    vel_mapper.scalar_range = 0.0, max_vel

    if 1:
        # Create a scalar bar
        if vertical_scale:
            scalar_bar = tvtk.ScalarBarActor(
                orientation="vertical", width=0.08, height=0.4
            )
        else:
            scalar_bar = tvtk.ScalarBarActor(
                orientation="horizontal", width=0.4, height=0.08
            )
        scalar_bar.title = "Speed (m/s)"
        scalar_bar.lookup_table = vel_mapper.lookup_table

        scalar_bar.property.color = 0.0, 0.0, 0.0  # black

        scalar_bar.title_text_property.color = 0.0, 0.0, 0.0
        scalar_bar.title_text_property.shadow = False

        scalar_bar.label_text_property.color = 0.0, 0.0, 0.0
        scalar_bar.label_text_property.shadow = False

        scalar_bar.position_coordinate.coordinate_system = "normalized_viewport"
        if vertical_scale:
            scalar_bar.position_coordinate.value = 0.01, 0.01, 0.0
        else:
            scalar_bar.position_coordinate.value = 0.1, 0.01, 0.0

        ren.add_actor(scalar_bar)

    imf = tvtk.WindowToImageFilter(input=rw, read_front_buffer="off")
    writer = tvtk.PNGWriter()

    ####################### end draw permanently on stuff ############################

    save_dt = 1.0 / save_fps

    if animate_path:
        data_dt = 1.0 / data_fps
        n_frames = len(obj_verts)
        dur = n_frames * data_dt
    else:
        data_dt = 0.0
        dur = 0.0

    print("data_fps", data_fps)
    print("data_dt", data_dt)
    print("save_fps", save_fps)

    t_now = 0.0
    frame_number = 0
    while t_now <= dur:
        frame_number += 1
        t_now += save_dt
        print("t_now", t_now)

        pos, ori = camera_animation_path.get_pos_ori(t_now)
        focal_point, view_up = pos_ori2fu(pos, ori)

        camera.position = tuple(pos)
        # camera.focal_point = (focal_point[0], focal_point[1], focal_point[2])
        # camera.view_up = (view_up[0], view_up[1], view_up[2])
        camera.focal_point = tuple(focal_point)
        camera.view_up = tuple(view_up)

        if data_dt != 0.0:
            draw_n_frames = int(round(t_now / data_dt))
        else:
            draw_n_frames = len(obj_verts)
        print("frame_number, draw_n_frames", frame_number, draw_n_frames)

        #################

        pd = tvtk.PolyData()
        pd.points = obj_verts[:draw_n_frames]
        real_frame_number = real_frames[:draw_n_frames][-1]
        pd.point_data.scalars = speeds
        if numpy.any(speeds > max_vel):
            print(
                "WARNING: maximum speed (%.3f m/s) exceeds color map max"
                % (speeds.max(),)
            )

        g = tvtk.Glyph3D(
            scale_mode="data_scaling_off", vector_mode="use_vector", input=pd
        )
        vel_mapper.input = g.output
        ss = tvtk.SphereSource(radius=options.radius)
        g.source = ss.output
        a = tvtk.Actor(mapper=vel_mapper)

        ##################

        ren.add_actor(a)

        if 1:
            imf.update()
            imf.modified()
            writer.input = imf.output
            # fname = 'movie_%s_%03d_frame%05d.png'%(filename_trimmed,obj_id,frame_number)
            fname = "movie_%s_%03d_frame%05d.png" % (
                filename_trimmed,
                obj_id,
                real_frame_number,
            )
            print("saving", fname)
            full_fname = os.path.join(output_dir, fname)
            writer.file_name = full_fname
            writer.write()

        ren.remove_actor(a)

    ren.add_actor(a)  # restore actors removed
    dur = dur + cam_only_move_duration

    while t_now < dur:
        frame_number += 1
        t_now += save_dt
        print("t_now", t_now)

        pos, ori = camera_animation_path.get_pos_ori(t_now)
        focal_point, view_up = pos_ori2fu(pos, ori)
        camera.position = tuple(pos)
        camera.focal_point = tuple(focal_point)
        camera.view_up = tuple(view_up)
        if 1:
            imf.update()
            imf.modified()
            writer.input = imf.output
            if len(use_obj_ids) == 1:
                fname = "movie_%s_%03d_frame%05d.png" % (
                    filename_trimmed,
                    obj_id,
                    frame_number,
                )
            else:
                fname = "movie_%s_many_frame%05d.png" % (filename_trimmed, frame_number)
            full_fname = os.path.join(output_dir, fname)
            writer.file_name = full_fname
            writer.write()

    if not is_mat_file:
        data_file.close()


def main():
    usage = "%prog FILE [options]"

    parser = OptionParser(usage)

    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        type="string",
        help="hdf5 file with data to display FILE",
        metavar="FILE",
    )

    parser.add_option("--obj-only", type="string")

    parser.add_option(
        "--draw-stim",
        type="string",
        dest="draw_stim_func_str",
        default="flydra_analysis.a2.conditions_draw:draw_default_stim",
    )

    parser.add_option(
        "--cam-only-move-duration",
        type="float",  # formerly called hover
        dest="cam_only_move_duration",
        default=5.0,
    )

    parser.add_option("--output-dir", type="string", dest="output_dir")

    parser.add_option("--animation-path-fname", type="string")

    parser.add_option(
        "--min-length",
        dest="min_length",
        type="int",
        help="minimum number of tracked points (not observations!) required to plot",
        default=10,
    )

    parser.add_option(
        "--radius",
        type="float",
        help="radius of line (in meters)",
        default=0.002,
        metavar="RADIUS",
    )

    parser.add_option(
        "--max-vel",
        type="string",
        help="maximum velocity of colormap",
        dest="max_vel",
        default="auto",
    )

    parser.add_option(
        "--disable-kalman-smoothing",
        action="store_false",
        dest="use_kalman_smoothing",
        default=True,
        help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)",
    )

    parser.add_option(
        "--vertical-scale",
        action="store_true",
        dest="vertical_scale",
        help="scale bar has vertical orientation",
    )

    parser.add_option(
        "--stim-xml",
        type="string",
        default=None,
        help="name of XML file with stimulus info",
    )

    parser.add_option(
        "--start", type="int", default=None,
    )

    parser.add_option(
        "--dynamic-model", type="string", default=None,
    )

    (options, args) = parser.parse_args()

    if options.filename is not None:
        args.append(options.filename)

    if len(args) > 1:
        print("arguments interpreted as FILE supplied more than once", file=sys.stderr)
        parser.print_help()
        return

    if len(args) < 1:
        parser.print_help()
        return

    h5_filename = args[0]

    if options.obj_only is not None:
        options.obj_only = options.obj_only.replace(",", " ")
        seq = map(int, options.obj_only.split())
        options.obj_only = seq

    if options.output_dir is None:
        options.output_dir = os.curdir

    doit(
        filename=h5_filename,
        obj_only=options.obj_only,
        cam_only_move_duration=options.cam_only_move_duration,
        use_kalman_smoothing=options.use_kalman_smoothing,
        min_length=options.min_length,
        vertical_scale=options.vertical_scale,
        draw_stim_func_str=options.draw_stim_func_str,
        max_vel=options.max_vel,
        floor=True,
        animation_path_fname=options.animation_path_fname,
        output_dir=options.output_dir,
        options=options,
    )


if __name__ == "__main__":
    main()
