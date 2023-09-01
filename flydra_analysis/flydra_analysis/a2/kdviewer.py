from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

if 1:
    # deal with old files, forcing to numpy
    import tables.flavor

    tables.flavor.restrict_flavors(keep=["numpy"])

import os, sys, math
import warnings

import pkg_resources
from tvtk.api import tvtk
from tvtk.common import configure_input_data

import numpy
import numpy as np
import tables as PT
from optparse import OptionParser
from . import core_analysis
import scipy.io
import datetime
import pkg_resources
import flydra_core.reconstruct as reconstruct
import flydra_analysis.analysis.result_utils as result_utils
import flydra_analysis.a2.xml_stimulus as xml_stimulus
import flydra_analysis.a2.flypos
import flydra_analysis.analysis.PQmath as PQmath

import cgtypes  # cgkit 1.x
import flydra_analysis.a2.pos_ori2fu

# from flydra_analysis.a2.experiment_layout import get_tvtk_actors_for_file
import flydra_analysis.version


def print_cam_props(camera):
    print("camera.parallel_projection = ", camera.parallel_projection)
    print("camera.focal_point = ", camera.focal_point)
    print("camera.position = ", camera.position)
    print("camera.view_angle = ", camera.view_angle)
    print("camera.view_up = ", camera.view_up)
    print("camera.clipping_range = ", camera.clipping_range)
    print("camera.parallel_scale = ", camera.parallel_scale)


def do_show_cameras(
    results, renderers, frustums=True, axes=True, labels=True, centers=True, length=2.0
):
    actors = []

    if isinstance(results, reconstruct.Reconstructor):
        R = results
    else:
        R = reconstruct.Reconstructor(results)

    if centers:
        cam_centers = tvtk.Points()

        for cam_id, pmat in R.Pmat.items():
            X = reconstruct.pmat2cam_center(pmat)  # X is column vector (matrix)
            X = numpy.array(X.flat)
            cam_centers.insert_next_point(*X)

        points_poly_data = tvtk.PolyData(points=cam_centers)
        ball = tvtk.SphereSource(radius=0.020, theta_resolution=25, phi_resolution=25)
        balls = tvtk.Glyph3D(
            scale_mode="data_scaling_off",
            vector_mode="use_vector",
            input=points_poly_data,
            source=ball.output,
        )
        mapBalls = tvtk.PolyDataMapper(input=balls.output)
        ballActor = tvtk.Actor(mapper=mapBalls)
        ballActor.property.diffuse_color = (1, 0, 0)
        ballActor.property.specular = 0.3
        ballActor.property.specular_power = 30
        actors.append(ballActor)

    if axes:
        for cam_id in R.Pmat.keys():
            pmat = R.get_pmat(cam_id)

            intrinsic_parameters, rotation_matrix = reconstruct.my_rq(pmat[:, :3])
            U = rotation_matrix[
                2, :
            ]  # 3rd row of rotation matrix (idea from drawscene.m in MultiCamSelfCal)
            U = U / math.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)  # normalize

            C = reconstruct.pmat2cam_center(pmat)  # column vector (matrix)
            C = C[:, 0]  # 1d array
            X = C + length * U

            verts = []
            lines = []

            verts.append(C)
            verts.append(X)
            lines.append([0, 1])

            pd = tvtk.PolyData()
            pd.points = verts
            pd.lines = lines
            pt = tvtk.TubeFilter(
                radius=0.001,
                input=pd,
                number_of_sides=4,
                vary_radius="vary_radius_off",
            )
            m = tvtk.PolyDataMapper(input=pt.output)

            a = tvtk.Actor(mapper=m)
            a.property.color = 0.9, 0.9, 0.9
            a.property.specular = 0.3
            actors.append(a)

    if frustums:
        line_points = tvtk.Points()
        polys = tvtk.CellArray()
        point_num = 0

        for cam_id in R.Pmat.keys():
            pmat = R.get_pmat(cam_id)
            width, height = R.get_resolution(cam_id)

            # cam center
            C = reconstruct.pmat2cam_center(pmat)  # X is column vector (matrix)
            C = numpy.array(C.flat)

            # cam orientation (used to select direction of ray)
            intrinsic_parameters, rotation_matrix = reconstruct.my_rq(pmat[:, :3])
            U = rotation_matrix[
                2, :
            ]  # 3rd row of rotation matrix (idea from drawscene.m in MultiCamSelfCal)
            U = U / math.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)  # normalize
            cam_axis = U

            # Note that this seems to only arbitrarily get direction
            # of ray (could be in front or behind camera).
            z = 1
            first_vert = None

            for x, y in (
                (0, 0),
                (0, height - 1),
                (width - 1, height - 1),
                (width - 1, 0),
            ):
                x2d = x, y, z
                X = R.find3d_single_cam(cam_id, x2d)  # returns column matrix
                X = X.flat
                X = X[:3] / X[3]

                line_points.insert_next_point(*C)
                point_num += 1

                U = X - C  # direction
                # rescale to unit length
                U = U / math.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)

                if 1:
                    # select direction closest to cam axis
                    U1 = U
                    U2 = -U
                    d1squared = numpy.sum((U1 - cam_axis) ** 2)
                    d2squared = numpy.sum((U2 - cam_axis) ** 2)
                    if d1squared < d2squared:
                        U = U1
                    else:
                        U = U2

                X = C + length * U

                line_points.insert_next_point(*X)
                point_num += 1

                if first_vert is None:
                    first_vert = point_num - 2
                else:
                    polys.insert_next_cell(4)
                    polys.insert_cell_point(point_num - 4)
                    polys.insert_cell_point(point_num - 3)
                    polys.insert_cell_point(point_num - 1)
                    polys.insert_cell_point(point_num - 2)

            polys.insert_next_cell(4)
            polys.insert_cell_point(point_num - 2)
            polys.insert_cell_point(point_num - 1)
            polys.insert_cell_point(first_vert + 1)
            polys.insert_cell_point(first_vert)

        profileData = tvtk.PolyData(points=line_points, polys=polys)
        profileMapper = tvtk.PolyDataMapper(input=profileData)
        profile = tvtk.Actor(mapper=profileMapper)
        p = profile.property
        p.opacity = 0.1
        p.diffuse_color = 1, 0, 0
        p.specular = 0.3
        p.specular_power = 30
        actors.append(profile)

    if labels:
        for cam_id, pmat in R.Pmat.items():
            X = reconstruct.pmat2cam_center(pmat)  # X is column vector (matrix)
            X = numpy.array(X.flat)

            ta = tvtk.TextActor(input=cam_id)
            # ta.set(scaled_text=True, height=0.05)
            pc = ta.position_coordinate
            pc.coordinate_system = "world"
            pc.value = X
            actors.append(ta)
    return actors


def set_color_for_obj_id(obj_id, a):
    if obj_id % 4 == 0:
        a.property.color = 0.9, 0.8, 0
    if obj_id % 4 == 1:
        a.property.color = 0, 0.45, 0.70
    if obj_id % 4 == 2:
        a.property.color = 0.3, 0.65, 0.10
    if obj_id % 4 == 3:
        a.property.color = 0, 1, 0


def doit(
    filename,
    show_obj_ids=False,
    start=None,
    stop=None,
    obj_start=None,
    obj_stop=None,
    obj_only=None,
    obj_filelist=None,
    show_n_longest=None,
    radius=0.002,  # in meters
    min_length=10,
    show_saccades=True,
    show_observations=False,
    draw_stim_func_str=None,
    use_kalman_smoothing=True,
    fps=None,
    up_dir=None,
    vertical_scale=False,
    max_vel=None,
    show_only_track_ends=False,
    save_still=False,
    exclude_vel_mps=None,
    exclude_vel_data="kalman",
    stereo=False,
    show_cameras=False,
    obj_color=False,
    link_all_simultaneous_objs=True,
    show_kalman_P=False,
    options=None,
):

    assert exclude_vel_data in [
        "kalman",
        "observations",
    ]  # kalman means smoothed or filtered, depending on use_kalman_smoothing

    all_max_vel = 0.0

    if up_dir is not None:
        up_dir = np.array(up_dir, dtype=np.float64)
        assert up_dir.shape == (3,)

    if link_all_simultaneous_objs:
        allsave = []

    if not use_kalman_smoothing:
        if options.dynamic_model is not None:
            print(
                "ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting dynamic model option (--dynamic-model)",
                file=sys.stderr,
            )
            sys.exit(1)
    dynamic_model_name = options.dynamic_model

    ca = core_analysis.get_global_CachingAnalyzer()
    obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(filename)
    if obj_ids is None:
        raise ValueError("no obj_ids in file")

    if options.stim_xml is not None:
        if data_file.filename.startswith("DATA") and (
            data_file.filename.endswith(".h5") or data_file.filename.endswith(".kh5")
        ):
            file_timestamp = data_file.filename[4:19]
        else:
            file_timestamp = None
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

    if not is_mat_file:

        mat_data = None
        if fps is None:
            fps = result_utils.get_fps(data_file, fail_on_error=False)

        if fps is None:
            fps = 100.0
            warnings.warn("Setting fps to default value of %f" % fps)

    if show_n_longest is not None:
        if obj_only is not None:
            raise ValueError("show_n_longest incompatible with --obj-only limiter")

        if len(use_obj_ids):
            print(
                "%d obj_ids total. Range is %d - %d"
                % (len(use_obj_ids), use_obj_ids[0], use_obj_ids[-1])
            )

        obj_ids_by_n_frames = {}
        for i, obj_id in enumerate(use_obj_ids):
            if (obj_start is not None) and (obj_id < obj_start):
                continue
            if (obj_stop is not None) and (obj_id > obj_stop):
                continue

            if i % 100 == 0:
                print("doing %d of %d (obj_id %d)" % (i, len(use_obj_ids), obj_id))

            if not ca.has_obj_id(obj_id, data_file):
                continue
            try:
                obs_rows = ca.load_dynamics_free_MLE_position(obj_id, data_file)
            except core_analysis.ObjectIDDataError as err:
                continue

            n_frames = int(obs_rows["frame"][-1]) - int(obs_rows["frame"][0]) + 1

            if exclude_vel_mps and exclude_vel_data == "kalman":
                try:
                    rows = ca.load_data(
                        obj_id,
                        data_file,
                        use_kalman_smoothing=use_kalman_smoothing,
                        dynamic_model_name=dynamic_model_name,
                        frames_per_second=fps,
                        up_dir=up_dir,
                        min_ori_quality_required=options.ori_qual,
                    )
                except core_analysis.NotEnoughDataToSmoothError:
                    warnings.warn(
                        "not enough data to smooth obj_id %d, skipping." % (obj_id,)
                    )
                    continue
                # central difference to find velocity
                velx = (rows["x"][2:] - rows["x"][:-2]) * 2 * fps
                vely = (rows["y"][2:] - rows["y"][:-2]) * 2 * fps
                velz = (rows["z"][2:] - rows["z"][:-2]) * 2 * fps
                vel_mag = numpy.sqrt(velx ** 2 + vely ** 2 + velz ** 2)
                median_vel_mag = numpy.median(vel_mag)
                if median_vel_mag < exclude_vel_mps:
                    continue

            obj_ids_by_n_frames.setdefault(n_frames, []).append(obj_id)

        n_frames_list = obj_ids_by_n_frames.keys()
        n_frames_list.sort()

        obj_only = []
        while len(n_frames_list):
            n_frames = n_frames_list.pop()
            obj_ids = obj_ids_by_n_frames[n_frames]
            obj_only.extend(obj_ids)
            if len(obj_only) >= show_n_longest:
                break

        print("longest traces = ", obj_only)
        use_obj_ids = numpy.array(obj_only)

    if obj_start is not None:
        use_obj_ids = use_obj_ids[use_obj_ids >= obj_start]
    if obj_stop is not None:
        use_obj_ids = use_obj_ids[use_obj_ids <= obj_stop]
    if obj_filelist is not None:
        obj_only = 1
    if obj_only is not None:
        if obj_filelist is not None:
            data = np.loadtxt(obj_filelist, delimiter="_")

            obj_only = np.array(data[:], dtype="int")

        use_obj_ids = numpy.array(obj_only)

    #################
    rw = tvtk.RenderWindow(
        size=(1024, 768), stereo_capable_window=stereo, alpha_bit_planes=True,
    )
    if stereo:
        ##     rw.stereo_render_on()
        ##     rw.set_stereo_type_to_red_blue()
        rw.set(stereo_type="red_blue", stereo_render=stereo)

    ## if show_obj_ids or (options.show_frames!=0):
    ##     # Because I can't get black text right now (despite trying),
    ##     # make background blue to see white text. - ADS
    ##     ren = tvtk.Renderer(background=(0.6,0.6,1.0)) # blue
    ## else:
    if 1:
        ren = tvtk.Renderer(background=(1.0, 1.0, 1.0))  # white

    camera = ren.active_camera
    actors = []
    actor2obj_id = {}

    if show_cameras:
        if is_mat_file:
            raise RuntimeError(".mat file does not contain camera information")
        actors.extend(do_show_cameras(extra["kresults"], [ren]))

    if 0:
        camera.parallel_projection = 0
        camera.focal_point = (
            0.52719625417063776,
            0.15695605837665305,
            0.10876143712478874,
        )
        camera.position = (
            0.39743071773877131,
            -0.4114652255728779,
            0.097431169175252269,
        )
        camera.view_angle = 30.0
        camera.view_up = (
            -0.072067516965519787,
            -0.0034285481144054573,
            0.99739386305323308,
        )
        camera.clipping_range = (0.25210456649736646, 1.0012868084455435)
        camera.parallel_scale = 0.294595461395
    if 0:
        camera.parallel_projection = 0
        camera.focal_point = (
            0.49827304637942593,
            0.20476671221773424,
            0.090222461715116345,
        )
        camera.position = (
            0.41982519417302594,
            -0.55501151899867784,
            0.40089956585064912,
        )
        camera.view_angle = 30.0
        camera.view_up = (
            0.025460553314687551,
            0.37610935779812088,
            0.92622541057865326,
        )
        camera.clipping_range = (0.38425211041324286, 1.3299558503823485)
        camera.parallel_scale = 0.294595461395
    if 1:
        camera.view_angle = 30.0
    if 0:
        camera.parallel_projection = 0
        camera.focal_point = [0.01268098, 0.07489683, 0.21150847]
        camera.position = [-0.22551933, -0.73470702, 1.59260368]
        camera.view_angle = 30.0
        camera.view_up = [0.21022155, 0.82717069, 0.5211483]
        camera.clipping_range = [1.03936226, 2.36410673]
        camera.parallel_scale = 1.0

    rw.add_renderer(ren)
    rwi = tvtk.RenderWindowInteractor(
        render_window=rw,
        interactor_style=tvtk.InteractorStyleTrackballCamera(),
        # stereo = stereo,
    )

    if options.lut is None:
        lut = tvtk.LookupTable(hue_range=(0.667, 0.0))
    elif options.lut in ("gray", "grey"):
        lut = tvtk.LookupTable(
            hue_range=(0, 0),
            saturation_range=(0, 0),
            # value_range = (0, .8), # don't go all the way to 1 to keep away from pure white
            value_range=(
                0.8,
                0,
            ),  # don't go all the way to 1 to keep away from pure white
        )

    #################

    if show_only_track_ends:
        track_end_verts = []

    if not len(use_obj_ids):
        raise ValueError("no trajectories to plot")

    had_any_obj_id_data = False
    obj_id2verts_frames = {}

    breakout = False
    for obj_id_enum, obj_id in enumerate(use_obj_ids):
        if breakout:
            break
        if (obj_id_enum % 100) == 0 and len(use_obj_ids) > 5:
            print("obj_id %d of %d" % (obj_id_enum, len(use_obj_ids)))
            if 0:
                import time

                now = time.time()
                if last_time is not None:
                    dur = now - last_time
                    print(dur, "seconds")
                last_time = now

        if not is_mat_file:
            # h5 file has timestamps for each frame
            # my_rows = ca.get_recarray(data_file,obj_id,which_data='kalman')
            if not options.fuse:
                # this is only useful for printing information
                try:
                    return_smoothed_directions = options.smooth_orientations
                    my_rows = ca.load_data(
                        obj_id,
                        data_file,
                        use_kalman_smoothing=use_kalman_smoothing,
                        dynamic_model_name=dynamic_model_name,
                        frames_per_second=fps,
                        up_dir=up_dir,
                        return_smoothed_directions=return_smoothed_directions,
                        min_ori_quality_required=options.ori_qual,
                    )
                except core_analysis.ObjectIDDataError as err:
                    continue

            if 0:
                my_timestamp = my_rows["timestamp"][0]
                dur = my_rows["timestamp"][-1] - my_timestamp
                print(
                    "%d 3D triangulation started at %s (took %.2f seconds)"
                    % (obj_id, datetime.datetime.fromtimestamp(my_timestamp), dur)
                )
                print(
                    "  estimate frames: %d - %d (%d frames)"
                    % (
                        my_rows["frame"][0],
                        my_rows["frame"][-1],
                        int(my_rows["frame"][-1]) - (my_rows["frame"][0]),
                    ),
                    end=" ",
                )
                if fps is None:
                    fpses = [60.0, 100.0, 200.0]
                else:
                    fpses = [fps]
                for my_fps in fpses:
                    print(
                        "(%.1f sec at %.1f fps)"
                        % (
                            (my_rows["frame"][-1] - my_rows["frame"][0]) / my_fps,
                            my_fps,
                        ),
                        end=" ",
                    )
                print()

        if show_observations:
            obs_rows, obs_directions = ca.load_dynamics_free_MLE_position(
                obj_id,
                data_file,
                with_directions=True,
                min_ori_quality_required=options.ori_qual,
            )

            if start is not None or stop is not None:
                obs_frames = obs_rows["frame"]
                ok1 = obs_frames >= start
                ok2 = obs_frames <= stop
                ok = ok1 & ok2
                obs_rows = obs_rows[ok]
                obs_directions = obs_directions[ok]

            obs_x = obs_rows["x"]
            obs_y = obs_rows["y"]
            obs_z = obs_rows["z"]
            obs_frames = obs_rows["frame"]
            if len(obs_frames):
                print("  observation frames: %d - %d" % (obs_frames[0], obs_frames[-1]))
            obs_X = numpy.vstack((obs_x, obs_y, obs_z)).T

            pd = tvtk.PolyData()
            pd.points = obs_X

            g = tvtk.Glyph3D(
                scale_mode="data_scaling_off", vector_mode="use_vector", input=pd
            )
            print("radius/3", radius / 3)
            ss = tvtk.SphereSource(
                radius=radius / 3,
                # theta_resolution=3,
                # phi_resolution=3,
            )
            g.source = ss.output
            vel_mapper = tvtk.PolyDataMapper(input=g.output)
            a = tvtk.Actor(mapper=vel_mapper)
            a.property.color = 1.0, 0.0, 0.0
            actors.append(a)
            actor2obj_id[a] = obj_id

            direction_length = 0.06  # 3 cm
            if options.show_observations_orientation and len(obs_directions):
                assert numpy.alltrue(PQmath.is_unit_vector(obs_directions))
                obs_directions = core_analysis.choose_orientations(
                    obs_rows,
                    obs_directions,
                    frames_per_second=fps,
                    elevation_up_bias_degrees=0,
                    up_dir=up_dir,
                )
                assert numpy.alltrue(PQmath.is_unit_vector(obs_directions))

                heads = obs_X + obs_directions * direction_length
                verts = numpy.vstack((heads, obs_X))

                tubes = [[i, i + len(heads)] for i in range(len(heads))]

                pd = tvtk.PolyData()
                pd.points = verts
                pd.lines = tubes

                pt = tvtk.TubeFilter(
                    radius=0.001,
                    input=pd,
                    number_of_sides=4,
                    vary_radius="vary_radius_off",
                )
                m = tvtk.PolyDataMapper(input=pt.output)
                a = tvtk.Actor(mapper=m)
                a.property.color = (1, 0, 0)  # red
                a.property.specular = 0.3
                actors.append(a)
                actor2obj_id[a] = obj_id

                del verts  # make sure it's not used below

        if options.fuse:
            print("fusing %s" % use_obj_ids)
            rows = flydra_analysis.a2.flypos.fuse_obj_ids(
                use_obj_ids,
                data_file,
                dynamic_model_name=dynamic_model_name,
                frames_per_second=fps,
            )
            breakout = True  # exit main loop after this run -- we're fusing all
        else:
            return_smoothed_directions = options.smooth_orientations
            rows = ca.load_data(
                obj_id,
                data_file,
                use_kalman_smoothing=use_kalman_smoothing,
                frames_per_second=fps,
                dynamic_model_name=dynamic_model_name,
                return_smoothed_directions=return_smoothed_directions,
                up_dir=up_dir,
                min_ori_quality_required=options.ori_qual,
            )

        if options.nth_frame != 1:
            cond = rows["frame"] % options.nth_frame == 0
            rows = rows[cond]

        if len(rows):
            frames = rows["frame"]
            n_frames = int(rows["frame"][-1]) - int(rows["frame"][0]) + 1
        else:
            n_frames = 0

        if n_frames < int(min_length):
            continue

        if not show_n_longest:  # already did this for show_n_longest above
            if exclude_vel_mps and exclude_vel_data == "kalman":
                # central difference to find velocity
                velx = (rows["x"][2:] - rows["x"][:-2]) * 2 * fps
                vely = (rows["y"][2:] - rows["y"][:-2]) * 2 * fps
                velz = (rows["z"][2:] - rows["z"][:-2]) * 2 * fps
                vel_mag = numpy.sqrt(velx ** 2 + vely ** 2 + velz ** 2)
                median_vel_mag = numpy.median(vel_mag)
                if median_vel_mag < exclude_vel_mps:
                    continue

        had_any_obj_id_data = True

        if not show_only_track_ends:
            frames = rows["frame"]
            if start is not None or stop is not None:
                ok1 = frames >= start
                ok2 = frames <= stop
                ok = ok1 & ok2
                rows = rows[ok]
                frames = rows["frame"]

            verts = numpy.array([rows["x"], rows["y"], rows["z"]]).T
            have_body_axis_information = "rawdir_x" in rows.dtype.fields
            if have_body_axis_information:
                if options.smooth_orientations:
                    verts_directions = numpy.array(
                        [rows["dir_x"], rows["dir_y"], rows["dir_z"]]
                    ).T
                else:
                    verts_directions = numpy.array(
                        [rows["rawdir_x"], rows["rawdir_y"], rows["rawdir_z"]]
                    ).T
            else:
                verts_directions = None
            obj_id2verts_frames[obj_id] = (verts, rows["frame"])

            if show_kalman_P:
                Ps_position = numpy.array([rows["P00"], rows["P11"], rows["P22"]]).T
                Ps_position = numpy.sqrt(
                    Ps_position
                )  # put in distance units, not variance units
            if link_all_simultaneous_objs:
                allsave.append(rows)
            if (
                not obj_color
                and options.highlight_start is None
                and options.highlight_stop is None
            ):
                if len(verts) >= 3:
                    verts_central_diff = verts[2:, :] - verts[:-2, :]
                    dt = 1.0 / fps
                    vels = verts_central_diff / (2 * dt)
                    speeds = numpy.sqrt(numpy.sum(vels ** 2, axis=1))
                    speeds = numpy.array(
                        [speeds[0]] + list(speeds) + [speeds[-1]]
                    )  # pad end points
                else:
                    speeds = numpy.zeros((verts.shape[1],))
                max_speed_this_obj = numpy.max(speeds)
                all_max_vel = max(all_max_vel, max_speed_this_obj)
                if max_vel is not None:
                    if max_speed_this_obj > max_vel:
                        print(
                            "WARNING: max_vel = %s, but max speed is %.2f"
                            % (max_vel, max_speed_this_obj)
                        )

        else:
            x0 = rows.field("x")[0]
            x1 = rows.field("x")[-1]

            y0 = rows.field("y")[0]
            y1 = rows.field("y")[-1]

            z0 = rows.field("z")[0]
            z1 = rows.field("z")[-1]

            track_end_verts.append((x0, y0, z0))
            track_end_verts.append((x1, y1, z1))

        if show_observations:
            if 0:
                # draw lines connecting observation with Kalman point

                line_verts = numpy.concatenate([verts, obs_X])

                line_edges = []
                for i, obs_frame in enumerate(obs_frames):
                    kidx = numpy.nonzero(rows["frame"] == obs_frame)[0]
                    if not len(kidx) == 1:
                        raise ValueError(
                            "length of kidx is not 1, it is %d" % (len(kidx),)
                        )
                    kidx = kidx[0]
                    line_edges.append([kidx, i + len(verts)])

                pd = tvtk.PolyData()
                pd.points = line_verts
                pd.lines = line_edges

                pt = tvtk.TubeFilter(
                    radius=0.001,
                    input=pd,
                    number_of_sides=4,
                    vary_radius="vary_radius_off",
                )
                m = tvtk.PolyDataMapper(input=pt.output)
                a = tvtk.Actor(mapper=m)
                a.property.color = (1, 0, 0)  # red
                a.property.specular = 0.3
                actors.append(a)

        if show_saccades:
            saccades = core_analysis.detect_saccades(rows, frames_per_second=fps,)
            saccade_verts = saccades["X"]

        #################

        if not show_only_track_ends:
            pd = tvtk.PolyData()
            pd.points = verts
            if (
                not obj_color
                and options.highlight_start is None
                and options.highlight_stop is None
            ):
                pd.point_data.scalars = speeds
            #            if numpy.any(speeds>max_vel):
            #                print 'WARNING: maximum speed (%.3f m/s) exceeds color map max'%(speeds.max(),)

            sphere_kw = dict(theta_resolution=10, phi_resolution=10,)
            if show_kalman_P:
                scale_mode = "scale_by_vector_components"
                pd.point_data.vectors = Ps_position
                # sphere_radius = radius*1e3
            else:
                scale_mode = "data_scaling_off"
                sphere_kw.update(dict(radius=radius))

            g = tvtk.Glyph3D(scale_mode=scale_mode, vector_mode="use_vector",)
            configure_input_data(g, pd)
            ss = tvtk.SphereSource(**sphere_kw)
            g.set_source_connection(ss.output_port)
            vel_mapper = tvtk.PolyDataMapper(input_connection=g.output_port)
            if not obj_color:
                vel_mapper.lookup_table = lut
                if options.highlight_start is None and options.highlight_stop is None:
                    if max_vel is not None:
                        vel_mapper.scalar_range = 0.0, float(max_vel)
                    else:
                        vel_mapper.scalar_range = 0.0, float(all_max_vel)
                else:
                    vel_mapper.scalar_range = 0.0, 1.0
            a = tvtk.Actor(mapper=vel_mapper)
            if show_observations:
                a.property.opacity = 0.3  # sets transparency/alpha
            if obj_color:
                set_color_for_obj_id(obj_id, a)
            elif (
                options.highlight_start is not None
                or options.highlight_stop is not None
            ):
                highlight = np.ones(rows["frame"].shape, dtype=np.bool_)
                if options.highlight_start is not None:
                    highlight &= rows["frame"] >= options.highlight_start
                if options.highlight_stop is not None:
                    highlight &= rows["frame"] <= options.highlight_stop
                pd.point_data.scalars = highlight.astype(np.float64)
            actors.append(a)
            actor2obj_id[a] = obj_id

            if verts_directions is not None and (
                options.smooth_orientations or options.body_axis
            ):
                smoothed_ori_verts = numpy.vstack(
                    (verts - (5 * radius * verts_directions), verts)
                )
                tubes = [[i, i + len(verts)] for i in range(len(verts))]

                pd = tvtk.PolyData()
                pd.points = smoothed_ori_verts
                pd.lines = tubes

                pt = tvtk.TubeFilter(
                    radius=radius * 0.4,
                    input=pd,
                    number_of_sides=10,
                    vary_radius="vary_radius_off",
                )
                m = tvtk.PolyDataMapper(input=pt.output)
                a = tvtk.Actor(mapper=m)
                if obj_color:
                    set_color_for_obj_id(obj_id, a)
                else:
                    a.property.color = (1, 0, 0)  # red
                a.property.specular = 0.3
                actors.append(a)
                actor2obj_id[a] = obj_id

        if show_obj_ids:
            if len(verts):
                print("showing obj_id %d at %s" % (obj_id, str(verts[0])))
                obj_id_ta = tvtk.TextActor(input=str(obj_id) + " start")
                obj_id_ta.text_property = tvtk.TextProperty(
                    color=(0.0, 0.0, 0.0),  # black
                )
                obj_id_ta.position_coordinate.coordinate_system = "world"
                obj_id_ta.position_coordinate.value = tuple(verts[0])
                actors.append(obj_id_ta)
                actor2obj_id[a] = obj_id
            else:
                print("no data for obj_id %d" % obj_id)

        if options.show_frames != 0:
            if len(frames):
                docond = (
                    (frames - options.show_frames_start) % options.show_frames
                ) == 0
                doframes = frames[docond] - options.show_frames_start
                doverts = verts[docond]

                for thisframe, thisvert in zip(doframes, doverts):
                    print("thisframe", thisframe)

                    obj_id_ta = tvtk.TextActor(
                        input="%d" % (thisframe * options.show_frames_gain,)
                    )
                    obj_id_ta.text_property = tvtk.TextProperty(
                        color=(0.0, 0.0, 0.0), shadow=True, font_size=20,  # black
                    )
                    obj_id_ta.position_coordinate.coordinate_system = "world"
                    obj_id_ta.position_coordinate.value = tuple(thisvert)
                    actors.append(obj_id_ta)
                    actor2obj_id[a] = obj_id

        ##################

        if show_saccades:
            pd = tvtk.PolyData()
            pd.points = saccade_verts

            g = tvtk.Glyph3D(
                scale_mode="data_scaling_off", vector_mode="use_vector", input=pd
            )
            ss = tvtk.SphereSource(
                radius=3 * radius, theta_resolution=20, phi_resolution=20,
            )
            g.source = ss.output
            mapper = tvtk.PolyDataMapper(input=g.output)
            a = tvtk.Actor(mapper=mapper)
            # a.property.color = (0,1,0) # green
            a.property.color = (0, 0, 0)  # black
            a.property.opacity = 0.3
            actors.append(a)
            actor2obj_id[a] = obj_id

    if link_all_simultaneous_objs:
        allsave = numpy.concatenate(allsave)
        allframes = allsave["frame"]
        allframes_unique = numpy.unique(allframes)
        link_verts = []
        link_edges = []
        vert_count = 0
        for frameno in allframes_unique:
            this_frame_data = allsave[allframes == frameno]
            if len(this_frame_data) < 2:
                continue

            start_vert = vert_count
            for this_row in this_frame_data:
                link_verts.append([this_row["x"], this_row["y"], this_row["z"]])
                vert_count += 1
            end_vert = vert_count
            link_edges.append(list(range(start_vert, end_vert)))

        pd = tvtk.PolyData()
        pd.points = link_verts
        pd.lines = link_edges

        pt = tvtk.TubeFilter(
            radius=0.001, input=pd, number_of_sides=4, vary_radius="vary_radius_off",
        )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = (1, 1, 1)
        a.property.specular = 0.3
        actors.append(a)

    ################################

    ## if draw_stim_func_str is None:
    ##     if 0:
    ##         draw_stim_func_str = 'default'
    ##     else:
    ##         # new style
    ##         stim_actors = get_tvtk_actors_for_file(filename=filename,
    ##                                                force_stimulus=options.force_stimulus,
    ##                                                )
    ##         actors.extend( stim_actors )

    if options.stim_xml is not None:
        if not is_mat_file:
            R = reconstruct.Reconstructor(data_file)
            if stim_xml.has_reconstructor():
                stim_xml.verify_reconstructor(R)

        actors.extend(stim_xml.get_tvtk_actors())

    if draw_stim_func_str:
        import flydra_analysis.a2.stim_plugins as stim_plugins

        plugin_loader = stim_plugins.PluginLoader()

        try:
            PluginClass = plugin_loader(draw_stim_func_str)
        except Exception as err:
            print("possible values for --draw-stim:")
            print(plugin_loader.all_names)
            raise

        plugin = PluginClass(filename=filename, force_stimulus=options.force_stimulus)
        stim_actors = plugin.get_tvtk_actors()
        actors.extend(stim_actors)

    ################################

    if not is_mat_file:
        # make sure this is after all uses of data_file
        extra["kresults"].close()

    if show_only_track_ends:
        pd = tvtk.PolyData()

        verts = numpy.array(track_end_verts)

        if 1:
            print("limiting ends shown to approximate arena boundaries")
            cond = (verts[:, 2] < 0.25) & (verts[:, 2] > -0.05)
            # cond = cond & (verts[:,1] < 0.29) & (verts[:,1] > 0.0)
            showverts = verts[cond]
        else:
            showverts = verts

        pd.points = showverts

        g = tvtk.Glyph3D(
            scale_mode="data_scaling_off", vector_mode="use_vector", input=pd
        )
        ss = tvtk.SphereSource(radius=0.005, theta_resolution=3, phi_resolution=3,)
        g.source = ss.output
        mapper = tvtk.PolyDataMapper(input=g.output)
        a = tvtk.Actor(mapper=mapper)
        # a.property.color = (0,1,0) # green
        a.property.color = (1, 0, 0)  # red
        a.property.opacity = 0.3
        actors.append(a)

    for a in actors:
        ren.add_actor(a)

    if options.show_axes:
        # Inspired by pyface.tvtk.decorated_scene
        marker = tvtk.OrientationMarkerWidget()

        axes = tvtk.AxesActor()
        axes.set(
            normalized_tip_length=(0.4, 0.4, 0.4),
            normalized_shaft_length=(0.6, 0.6, 0.6),
            shaft_type="cylinder",
        )

        if 1:
            p = axes.x_axis_caption_actor2d.caption_text_property
            axes.y_axis_caption_actor2d.caption_text_property = p
            axes.z_axis_caption_actor2d.caption_text_property = p

            p.color = 0.0, 0.0, 0.0  # black
            p.font_size = 40

        marker.orientation_marker = axes
        marker.interactor = rwi
        marker.enabled = True

    if (
        not show_only_track_ends
        and not obj_color
        and had_any_obj_id_data
        and options.highlight_start is None
        and options.highlight_stop is None
    ):
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

        if 1:
            # Use the ScalarBarWidget so we can drag the scalar bar around.
            sc_bar_widget = tvtk.ScalarBarWidget(
                interactor=rwi, scalar_bar_actor=scalar_bar
            )

            rwi.initialize()
            sc_bar_widget.enabled = True

    if 1:
        picker = tvtk.CellPicker(tolerance=1e-9)
        # print 'dir(picker)',dir(picker)
        def annotatePick(object, event):
            # XXX keep all this math for reference with pos_ori2fu.py
            vtm = numpy.array(
                [
                    [
                        ren.active_camera.view_transform_matrix.get_element(i, j)
                        for j in range(4)
                    ]
                    for i in range(4)
                ]
            )
            # print 'camera.view_transform_matrix = ',vtm
            vtmcg = cgtypes.mat4(list(vtm.T))
            view_translation, view_rotation_mat4, view_scaling = vtmcg.decompose()
            q = aa = cgtypes.quat().fromMat(view_rotation_mat4)
            # print 'orientation quaternion',q
            aa = q.toAngleAxis()
            # print
            # print 'camera.position = ',ren.active_camera.position
            cpos = (
                view_rotation_mat4.inverse() * -view_translation
            )  # same as camera.position
            # print 'view_rotation_mat4.inverse()*-view_translation',cpos
            # print 'view_scaling',view_scaling

            # print 'camera.orientation_wxyz = ',ren.active_camera.orientation_wxyz
            # print 'aa',aa
            # print 'q',q
            # print 'camera.focal_point = ',ren.active_camera.focal_point
            # print 'camera.view_up = ',ren.active_camera.view_up

            upq = view_rotation_mat4.inverse() * cgtypes.vec3(0, 1, 0)  # get view up
            vd = view_rotation_mat4.inverse() * cgtypes.vec3(
                0, 0, -1
            )  # get view forward
            # print 'upq',upq
            # print 'viewdir1',(vd).normalize()
            # print 'viewdir2',(cgtypes.vec3(ren.active_camera.focal_point)-cpos).normalize()
            # print
            p = cgtypes.vec3(camera.position)
            print("animation path variable (t=time) (t,x,y,z,qw,qx,qy,qz):")
            print("t", p[0], p[1], p[2], q.w, q.x, q.y, q.z)
            print()

            if not picker.cell_id < 0:
                found = set([])
                for actor in picker.actors:
                    objid = actor2obj_id[actor]
                    found.add(objid)
                found = list(found)
                found.sort()

                # find frame for each point found
                for objid in found:
                    verts, this_obj_frames = obj_id2verts_frames[objid]
                    dists3d = verts - picker.pick_position
                    dists = numpy.sum(dists3d ** 2, axis=1)
                    idx = numpy.argmin(dists)
                    print(
                        "obj_id %d, frame %d, closest vert: %s"
                        % (objid, this_obj_frames[idx], verts[idx])
                    )

            if 1:
                imf = tvtk.WindowToImageFilter(
                    input=rw, input_buffer_type="rgba", read_front_buffer="off"
                )
                writer = tvtk.PNGWriter()

                imf.update()
                imf.modified()
                writer.input = imf.output
                fname = "kdviewer_output.png"
                writer.file_name = fname
                writer.write()

        picker.add_observer("EndPickEvent", annotatePick)
        rwi.picker = picker

    if not save_still:
        rwi.start()
        print_cam_props(ren.active_camera)
    else:
        imf = tvtk.WindowToImageFilter(
            input=rw, input_buffer_type="rgba", read_front_buffer="off"
        )
        writer = tvtk.PNGWriter()

        imf.update()
        imf.modified()
        writer.input = imf.output
        fname = "kdviewer_output.png"
        writer.file_name = fname
        writer.write()


def main():
    usage = "%prog FILE [options]"

    # A man page can be generated with:
    # 'help2man -N -n kdviewer kdviewer > kdviewer.1'

    parser = OptionParser(usage)

    parser.add_option(
        "--version",
        action="store_true",
        dest="version",
        help="print version and quit",
        default=False,
    )

    parser.add_option(
        "-f",
        "--file",
        dest="filename",
        type="string",
        help="hdf5 file with data to display FILE",
        metavar="FILE",
    )

    ##    parser.add_option("--debug", type="int",
    ##                      help="debug level",
    ##                      metavar="DEBUG")

    parser.add_option(
        "--start", type="int", help="first frame to plot", metavar="START"
    )

    parser.add_option("--stop", type="int", help="last frame to plot", metavar="STOP")

    parser.add_option("--highlight-start", type="int")

    parser.add_option("--highlight-stop", type="int")

    parser.add_option(
        "--obj-start",
        dest="obj_start",
        type="int",
        help="first object ID to plot",
        metavar="OBJSTART",
    )

    parser.add_option(
        "--obj-stop",
        dest="obj_stop",
        type="int",
        help="last object ID to plot",
        metavar="OBJSTOP",
    )

    parser.add_option("--obj-only", type="string")

    parser.add_option(
        "--obj-filelist",
        type="string",
        help="pull list of obj ids from a text file where first column is the object ids",
    )

    parser.add_option("--up-dir", type="string")

    parser.add_option(
        "--draw-stim",
        type="string",
        dest="draw_stim_func_str",
        default=None,
        help=(
            "DEPRECATED. name of drawing plugin. use a "
            "non-existant name to print list of "
            "availabe names"
        ),
    )

    parser.add_option(
        "--stim-xml",
        type="string",
        default=None,
        help="name of XML file with stimulus info",
    )

    parser.add_option(
        "--dynamic-model", type="string", default=None,
    )

    parser.add_option(
        "--lut", type="string", help="lookup table (e.g. 'gray')", default=None,
    )

    parser.add_option("--n-top-traces", type="int", help="show N longest traces")

    parser.add_option(
        "--min-length",
        dest="min_length",
        type="int",
        help=(
            "minimum number of tracked points " "(not observations!) required to plot"
        ),
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
        type="float",
        help="maximum velocity of colormap",
        dest="max_vel",
        default=None,
    )

    parser.add_option(
        "--exclude-vel",
        type="float",
        help=("exclude traces with median velocity less than " "this value"),
        dest="exclude_vel_mps",
        default=None,
    )

    parser.add_option(
        "--show-obj-ids",
        action="store_true",
        dest="show_obj_ids",
        help="show object ID numbers at start of trajectory",
    )

    parser.add_option(
        "--disable-axes",
        action="store_false",
        dest="show_axes",
        default=True,
        help="disable showing axies",
    )

    parser.add_option(
        "--nth-frame",
        type="int",
        default=1,
        help="show every nth frame (0=no frame numbers)",
    )

    parser.add_option(
        "--show-frames",
        type="int",
        default=0,
        help="show frame number interval (0=no frame numbers)",
    )

    parser.add_option(
        "--show-frames-gain",
        type="float",
        default=1.0,
        help="show frame number interval (0=no frame numbers)",
    )

    parser.add_option(
        "--show-frames-start", type="int", default=0, help="show frame number start"
    )

    parser.add_option(
        "--show-saccades",
        action="store_true",
        dest="show_saccades",
        help="show saccades",
    )

    parser.add_option(
        "--show-only-track-ends", action="store_true", dest="show_only_track_ends"
    )

    parser.add_option(
        "--show-observations", action="store_true", help="show observations"
    )

    parser.add_option(
        "--show-observations-orientation",
        action="store_true",
        help="show body axis observations",
    )

    parser.add_option(
        "--stereo",
        action="store_true",
        dest="stereo",
        help="display in stereo (red-blue analglyphic)",
        default=False,
    )

    parser.add_option(
        "--show-cameras",
        action="store_true",
        help="show camera locations/frustums",
        default=False,
    )

    parser.add_option(
        "--fps",
        dest="fps",
        type="float",
        default=None,
        help=("frames per second (used for Kalman " "filtering/smoothing)"),
    )

    parser.add_option(
        "--disable-kalman-smoothing",
        action="store_false",
        dest="use_kalman_smoothing",
        default=True,
        help=(
            "show original, causal Kalman filtered data "
            "(rather than Kalman smoothed observations)"
        ),
    )

    parser.add_option(
        "--show-kalman-P",
        action="store_true",
        dest="show_kalman_P",
        default=False,
        help="show Kalman P",
    ),

    parser.add_option(
        "--vertical-scale",
        action="store_true",
        dest="vertical_scale",
        help="scale bar has vertical orientation",
    )

    parser.add_option(
        "--save-still",
        action="store_true",
        dest="save_still",
        help="save still image as kdviewer_output.png",
    )

    parser.add_option(
        "--obj-color", action="store_true", default=False, dest="obj_color"
    )

    parser.add_option(
        "--link-all-simultaneous-objs",
        action="store_true",
        default=False,
        dest="link_all_simultaneous_objs",
    )

    parser.add_option(
        "--force-stimulus",
        action="store_true",
        help="raise error if stimulus condition not found",
        default=False,
    )

    parser.add_option(
        "--smooth-orientations",
        action="store_true",
        help="display smoothed orientations",
        default=False,
    )

    parser.add_option(
        "--body-axis",
        action="store_true",
        help="show body axis (orientation) data if available",
        default=False,
    )

    parser.add_option(
        "--fuse",
        action="store_true",
        help=(
            "fuse obj_ids specified in fanout .xml file into " "one contiguous trace"
        ),
        default=False,
    )

    parser.add_option(
        "--ori-qual",
        type="float",
        default=None,
        help=("minimum orientation quality to use"),
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
        options.obj_only = core_analysis.parse_seq(options.obj_only)

        if options.obj_start is not None or options.obj_stop is not None:
            raise ValueError("cannot specify start and stop with --obj-only option")

    if options.up_dir is not None:
        up_dir = core_analysis.parse_seq(options.up_dir)
    else:
        up_dir = None

    if options.version:
        print("kdviewer %s" % (flydra_analysis.version.__version__,))

    doit(
        filename=h5_filename,
        start=options.start,
        stop=options.stop,
        obj_start=options.obj_start,
        obj_stop=options.obj_stop,
        obj_only=options.obj_only,
        obj_filelist=options.obj_filelist,
        use_kalman_smoothing=options.use_kalman_smoothing,
        show_n_longest=options.n_top_traces,
        show_obj_ids=options.show_obj_ids,
        radius=options.radius,
        min_length=options.min_length,
        show_saccades=options.show_saccades,
        show_observations=options.show_observations,
        draw_stim_func_str=options.draw_stim_func_str,
        fps=options.fps,
        vertical_scale=options.vertical_scale,
        max_vel=options.max_vel,
        show_only_track_ends=options.show_only_track_ends,
        save_still=options.save_still,
        exclude_vel_mps=options.exclude_vel_mps,
        stereo=options.stereo,
        show_cameras=options.show_cameras,
        obj_color=options.obj_color,
        link_all_simultaneous_objs=options.link_all_simultaneous_objs,
        show_kalman_P=options.show_kalman_P,
        options=options,
        up_dir=up_dir,
    )


if __name__ == "__main__":
    main()
