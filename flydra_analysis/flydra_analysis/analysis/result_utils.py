from __future__ import print_function
import tables as PT

# pytables files stored using Numeric would by default return Numeric-based results.
# We want to force those results to be returned as numpy recarrays.
# Note that we need to keep "python" in the flavors list, otherwise
# pytables breaks.
import tables.flavor

tables.flavor.restrict_flavors(keep=["python", "numpy"])  # ensure pytables 2.x


import numpy as np
import sys, os, re, hashlib
import json

import warnings
from distutils.version import StrictVersion

import datetime
import pytz  # from http://pytz.sourceforge.net/
import pytest

DEFAULT_TZ = "US/Pacific"

# should avoid using any matplotlib here -- we want to keep this
# module lean and mean


def md5sum_headtail(filename):
    """quickly calculate a hash value for an even giant file"""
    fd = open(filename, mode="rb")
    start_bytes = fd.read(1000)

    try:
        fd.seek(-1000, os.SEEK_END)
    except IOError as err:
        # it's OK, we'll just read up to another 1000 bytes
        pass

    stop_bytes = fd.read(1000)
    bytes = start_bytes + stop_bytes
    m = hashlib.md5()
    m.update(bytes)
    return m.hexdigest()


def status(status_string):
    print(" status:", status_string)
    sys.stdout.flush()


def get_camn(results, cam, remote_timestamp=None, frame=None):
    """helper function to get camn given timestamp or frame number

    last used 2006-05-17

    """
    if not isinstance(cam, str):
        camn = cam
        return camn

    this_cam_id = cam
    possible_camns = []
    for row in results.root.cam_info:
        if row["cam_id"] == this_cam_id:
            possible_camns.append(row["camn"])

    table = results.root.data2d_camera_summary
    for row in table.where("cam_id==this_cam_id"):
        camn = None
        if row["camn"] in possible_camns:
            if remote_timestamp is not None:
                if row["start_timestamp"] <= remote_timestamp <= row["stop_timestamp"]:
                    if camn is not None:
                        if camn != row["camn"]:
                            raise RuntimeError(
                                "Found camn already! (Is frame from different run than timestamp?)"
                            )
                    camn = row["camn"]
            if frame is not None:
                if row["start_frame"] <= frame <= row["stop_frame"]:
                    if camn is not None:
                        if camn != row["camn"]:
                            raise RuntimeError(
                                "Found camn already! (Is frame from different run than timestamp?)"
                            )
                    camn = row["camn"]
    if camn is None:
        raise RuntimeError("could not find frame or timestamp")
    return camn


def get_frame_from_camn_and_timestamp(results, camn, remote_timestamp):
    """helper function

    last used 2006-06-06
    """
    found = False
    data2d = results.root.data2d_distorted
    if PT.__version__ <= "1.3.2":
        # if type(remote_timestamp)==np.float64scalar:
        remote_timestamp = float(remote_timestamp)
    for row in data2d.where(data2d.cols.timestamp == remote_timestamp):
        test_camn = row["camn"]
        if test_camn == camn:
            frame = row["frame"]
            found = True
            break
    if not found:
        raise ValueError("No data found for cam and remote_timestamp")
    return frame


def get_camn_and_frame(results, cam, remote_timestamp):
    """helper function

    last used 2006-06-06
    """
    camn = get_camn(results, cam, remote_timestamp=remote_timestamp)
    frame = get_frame_from_camn_and_timestamp(results, camn, remote_timestamp)
    return camn, frame


def get_camn_and_remote_timestamp(results, cam, frame):
    """helper function

    last used 2006-05-08
    """
    camn = get_camn(results, cam, frame=frame)
    found = False
    data2d = results.root.data2d_distorted
    try:
        for row in data2d.where(data2d.cols.frame == frame):
            test_camn = row["camn"]
            if test_camn == camn:
                timestamp = row["timestamp"]
                found = True
                break
    except TypeError:
        print("frame", frame)
        print("repr(frame)", repr(frame))
        print("type(frame)", type(frame))
        raise
    if not found:
        raise ValueError("No data found for cam and frame")
    return camn, timestamp


def get_cam_ids(results):
    cam_info = results.root.cam_info
    cam_ids = list(set(cam_info.cols.cam_id))
    cam_ids.sort()
    return cam_ids


def get_caminfo_dicts(results):
    """
    camn2cam_id, cam_id2camns = get_caminfo_dicts(results)
    """
    # camera info
    cam_info = results.root.cam_info
    cam_id2camns = {}
    camn2cam_id = {}

    for row in cam_info:
        cam_id, camn = row["cam_id"], row["camn"]
        cam_id = cam_id.strip()  # workaround pytables 1.3 save bug
        if sys.version_info.major >= 3:
            cam_id = cam_id.decode("utf-8")
        cam_id2camns.setdefault(cam_id, []).append(camn)
        camn2cam_id[camn] = cam_id
    return camn2cam_id, cam_id2camns


def get_results(filename, mode="r+", create_camera_summary=False):
    h5file = PT.open_file(filename, mode=mode)
    if hasattr(h5file.root, "data3d_best"):
        frame_col = h5file.root.data3d_best.cols.frame
        if frame_col.index is None:
            print("creating index on data3d_best.cols.frame ...")
            frame_col.create_index()
            print("done")

    if False and hasattr(h5file.root, "data2d"):
        frame_col = h5file.root.data2d.cols.frame
        if frame_col.index is None:
            print("creating index on data2d.cols.frame ...")
            frame_col.create_index()
            print("done")

    ##        timestamp_col = h5file.root.data2d.cols.timestamp
    ##        if timestamp_col.index is None:
    ##            print 'creating index on data2d.cols.timestamp ...'
    ##            timestamp_col.create_index()
    ##            print 'done'

    if hasattr(h5file.root, "data2d_distorted"):
        ##        timestamp_col = h5file.root.data2d_distorted.cols.timestamp
        ##        if timestamp_col.index is None:
        ##            print 'creating index on data2d_distorted.cols.timestamp ...'
        ##            timestamp_col.create_index()
        ##            print 'done'
        pass

    if create_camera_summary and not hasattr(h5file.root, "data2d_camera_summary"):
        if not hasattr(h5file.root, "data2d_distorted"):
            raise ValueError("need data2d_distorted to make camera summary")
        print("creating data2d camera summary ...")
        create_data2d_camera_summary(h5file)
        print("done")
    return h5file


def get_f_xyz_L_err(results, max_err=10, typ="best", include_timestamps=False):
    """workhorse function to get 3D data from file

    returns:
    (f,X,L,err)
    if include_timestamps is True:
    (f,X,L,err,timestamps)

    where:
    f is frame numbers
    X is 3D position coordinates
    L is Pluecker line coordinates
    err is mean reprojection distance
    timestamps are the timestamps on the 3D reconstruction computer

    last used 2006-05-16
    """
    if typ == "fast":
        data3d = results.root.data3d_fast
    elif typ == "best":
        data3d = results.root.data3d_best

    if max_err is not None:
        f = []
        x = []
        y = []
        z = []
        xyz = []
        L = []
        err = []
        timestamps = []
        for row in data3d.where("mean_dist <= max_err"):
            f.append(row["frame"])
            xyz.append((row["x"], row["y"], row["z"]))
            L.append((row["p0"], row["p1"], row["p2"], row["p3"], row["p4"], row["p5"]))
            err.append(row["mean_dist"])
            timestamps.append(row["timestamp"])
        f = np.array(f)
        xyz = np.array(xyz)
        L = np.array(L)
        err = np.array(err)
        timestamps = np.array(timestamps)
    else:
        frame_col = data3d.cols.frame
        if not len(frame_col):
            print("no 3D data")
            return
        f = np.array(frame_col)
        timestamps = np.array(timestamps)
        x = np.array(data3d.cols.x)
        y = np.array(data3d.cols.y)
        z = np.array(data3d.cols.z)
        xyz = np.concatenate(
            (x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), axis=1
        )
        p0 = np.array(data3d.cols.p0)[:, np.newaxis]
        p1 = np.array(data3d.cols.p1)[:, np.newaxis]
        p2 = np.array(data3d.cols.p2)[:, np.newaxis]
        p3 = np.array(data3d.cols.p3)[:, np.newaxis]
        p4 = np.array(data3d.cols.p4)[:, np.newaxis]
        p5 = np.array(data3d.cols.p5)[:, np.newaxis]
        L = np.concatenate((p0, p1, p2, p3, p4, p5), axis=1)
        err = np.array(data3d.cols.mean_dist)

    if hasattr(results.root, "ignore_frames"):
        good = np.argsort(f)
        good_set = set(np.argsort(f))
        for row in results.root.ignore_frames:
            start_frame, stop_frame = row["start_frame"], row["stop_frame"]
            head = np.where(f < start_frame)
            tail = np.where(f > stop_frame)
            head_set = set(head[0])
            tail_set = set(tail[0])

            good_set = (good_set & head_set) | (good_set & tail_set)
        good_idx = list(good_set)
        good_idx.sort()
    else:
        good_idx = np.argsort(f)

    f = np.take(f, good_idx, axis=0)
    xyz = np.take(xyz, good_idx, axis=0)
    L = np.take(L, good_idx, axis=0)
    err = np.take(err, good_idx, axis=0)
    timestamps = np.take(timestamps, good_idx, axis=0)

    rval = [f, xyz, L, err]
    if include_timestamps:
        rval.append(timestamps)
    return tuple(rval)


def get_reconstructor(results):
    import flydra_core.reconstruct

    return flydra_core.reconstruct.Reconstructor(results)


def get_resolution(results, cam_id):
    return tuple(results.root.calibration.resolution.__getattr__(cam_id))


def create_data2d_camera_summary(results):
    class Data2DCameraSummary(PT.IsDescription):
        cam_id = PT.StringCol(16, pos=0)
        camn = PT.Int32Col(pos=1)
        start_frame = PT.UInt64Col(pos=2)
        stop_frame = PT.UInt64Col(pos=3)
        start_timestamp = PT.FloatCol(pos=4)
        stop_timestamp = PT.FloatCol(pos=5)

    data2d = results.root.data2d_distorted  # make sure we have 2d data table
    camn2cam_id, cam_id2camns = get_caminfo_dicts(results)
    table = results.create_table(
        results.root,
        "data2d_camera_summary",
        Data2DCameraSummary,
        "data2d camera summary",
    )
    for camn in camn2cam_id:
        cam_id = camn2cam_id[camn]
        print("creating 2d camera index for camn %d, cam_id %s" % (camn, cam_id))

        first_row = True
        this_camn = camn
        for row_data2d in data2d.where("camn == this_camn"):
            ts = row_data2d["timestamp"]
            f = row_data2d["frame"]
            if first_row:
                start_timestamp = ts
                stop_timestamp = ts
                start_frame = f
                stop_frame = f
                first_row = False
            start_timestamp = min(start_timestamp, ts)
            stop_timestamp = max(stop_timestamp, ts)
            start_frame = min(start_frame, f)
            stop_frame = max(stop_frame, f)
        newrow = table.row
        newrow["cam_id"] = cam_id
        newrow["camn"] = camn
        newrow["start_frame"] = start_frame
        newrow["stop_frame"] = stop_frame
        newrow["start_timestamp"] = start_timestamp
        newrow["stop_timestamp"] = stop_timestamp
        newrow.append()
    table.flush()


def timestamp2string(ts_float, timezone=DEFAULT_TZ):
    if isinstance(timezone, datetime.tzinfo):
        tz = timezone
    else:
        tz = pytz.timezone(timezone)
    dt_ts = datetime.datetime.fromtimestamp(ts_float, tz)
    # dt_ts.ctime()
    return dt_ts.isoformat()


def model_remote_to_local(remote_timestamps, local_timestamps, debug=False):
    """for timestamps"""
    a1 = remote_timestamps[:, np.newaxis]
    a2 = np.ones((len(remote_timestamps), 1))
    A = np.hstack((a1, a2))
    b = local_timestamps[:, np.newaxis]
    x, resids, rank, s = np.linalg.lstsq(A, b)
    if debug:
        print(
            "in model_remote_to_local: N=%d, resids=%s"
            % (len(remote_timestamps), resids)
        )
    gain = x[0, 0]
    offset = x[1, 0]
    return gain, offset


class TimeModel:
    def __init__(self, gain, offset):
        self.gain = gain
        self.offset = offset

    def timestamp2framestamp(self, mainbain_timestamp):
        return (mainbain_timestamp - self.offset) / self.gain

    def framestamp2timestamp(self, framestamp):
        return framestamp * self.gain + self.offset


def frame2timestamp_command():
    h5_filename, frame_str = sys.argv[1:3]
    frame = int(frame_str)
    assert len(sys.argv) == 3
    results = tables.open_file(h5_filename, mode="r")
    model = get_time_model_from_data(results)
    print(repr(model.framestamp2timestamp(frame)))
    results.close()


def timestamp2frame_command():
    h5_filename, timestamp_str = sys.argv[1:3]
    timestamp = float(timestamp_str)
    assert len(sys.argv) == 3
    results = tables.open_file(h5_filename, mode="r")
    model = get_time_model_from_data(results)
    print(repr(model.timestamp2framestamp(timestamp)))
    results.close()


class NoTimestampDataError(Exception):
    pass


class TextlogParseError(Exception):
    pass


def read_textlog_header(results, fail_on_error=True):
    try:
        textlog1 = results.root.textlog.read_coordinates([0])
    except PT.exceptions.NoSuchNodeError as err:
        if fail_on_error:
            raise
        else:
            return None
    infostr = textlog1["message"].tostring().strip(b"\x00").decode("utf-8")
    if not (
        infostr.startswith("MainBrain running at")
        or infostr.startswith("kalmanize running at")
        or infostr.startswith("retrack_reuse_data_association running at")
    ):
        raise TextlogParseError("could not parse textlog - old version?")
    fps_str = infostr.split()[3]
    parsed = {}
    parsed["fps"] = fps_str
    re_paren = re.compile(r".*\((.*)\)")
    paren_contents = re_paren.search(infostr)
    paren_contents = paren_contents.groups()[0]
    paren_contents = paren_contents.split(",")
    for pc in paren_contents:
        elements = pc.strip().split()
        name = elements.pop(0)
        strvalue = ' '.join(elements)
        if strvalue == "unknown":
            continue
        parsed[name] = strvalue

    # Backwards compatibility to get version number in old files.
    if "flydra_version" not in parsed:
        # get second line of text log, if it exists
        try:
            textlog2 = results.root.textlog.read_coordinates([1])
        except IndexError as err:
            pass
        else:
            if textlog2["mainbrain_timestamp"] == textlog1["mainbrain_timestamp"]:
                assert textlog2["cam_id"] == b"mainbrain"
                message2 = textlog2["message"].tostring().strip(b"\x00").decode("utf-8")
                version_start_string = "using flydra version "
                if message2.startswith(version_start_string):
                    test_version = message2[len(version_start_string) :]
                    try:
                        # check if it can be parsed
                        StrictVersion(test_version)
                    except ValueError as err:
                        # cannot be parsed
                        pass
                    else:
                        parsed["flydra_version"] = test_version
    return parsed


def calc_fps_from_data(results, field_name):
    row0 = None
    idx = 0
    while 1:
        row = results.root.data2d_distorted.read_coordinates([idx])
        idx += 1
        if not np.isnan(row[field_name]):
            row0 = row
            break

    row1 = None
    idx = -1
    while 1:
        row = results.root.data2d_distorted.read_coordinates([idx])
        idx -= 1
        if not np.isnan(row[field_name]):
            row1 = row
            break

    frame0 = row0["frame"][0]
    frame1 = row1["frame"][0]
    t0 = row0[field_name][0]
    t1 = row1[field_name][0]

    time_range = t1 - t0
    frame_range = frame1 - frame0
    fps = frame_range / time_range

    return fps


def get_fps(results, fail_on_error=True):
    parsed = read_textlog_header(results, fail_on_error=fail_on_error)
    if parsed is None and not fail_on_error:
        return None
    result = float(parsed["fps"])

    if np.isnan(result):
        # calculate fps from data in case correct fps was not saved
        d2d = results.root.data2d_distorted[:10000]
        nnz = np.sum(~np.isnan(d2d["timestamp"]))
        if nnz > 10:
            # prefer 'timestamp' field, which can be nan if time model was not present
            result = calc_fps_from_data(results, field_name="timestamp")
        else:
            # else use 'cam_received_timestamp', which is always present
            result = calc_fps_from_data(results, field_name="cam_received_timestamp")

    if fail_on_error and np.isnan(result):
        raise ValueError("nan is not a valid frames per second value")
    return result


def get_tzname0(results, default=DEFAULT_TZ):
    parsed = read_textlog_header(results, fail_on_error=True)
    return parsed.get("time_tzname0", default)


def get_tz(results, default=DEFAULT_TZ):
    tzname0 = get_tzname0(results, default=default)
    tz = pytz.timezone(tzname0)
    return tz


def get_time_model_from_data(results, debug=False, full_output=False):
    parsed = read_textlog_header(results)

    flydra_version = parsed.get("flydra_version", "0.0.0")

    if "top" in parsed:
        # get the timer top value
        timer_max = int(parsed["top"])
    elif StrictVersion(flydra_version) >= StrictVersion("0.5.3"):
        timer_max = 255
    else:
        return None

    if debug:
        print('I found the timer max ("top") to be %d.' % timer_max)
        FOSC = 8000000  # 8 MHz
        CS_all = [1, 8, 64, 256, 1024]
        CS_known = False
        if "trigger_CS3" in parsed:
            CS = int(parsed["trigger_CS3"])
            CS_all = [CS]
            CS_known = True
        for CS in CS_all:
            F_CLK = FOSC / float(CS)  # clock frequency, Hz
            clock_tick_duration = 1.0 / F_CLK
            carrier_duration = timer_max * clock_tick_duration
            carrier_freq = 1.0 / carrier_duration
            if CS_known:
                print("  (%.1f Hz, CS=%s)" % (float(carrier_freq), CS))
            else:
                print("  (%.1f Hz if CS=%s)" % (float(carrier_freq), CS))

    # open the log of at90usb clock info

    node_type = results.root._g_get_objinfo("trigger_clock_info")
    if node_type == "NoSuchNode":
        # No time model possible
        return

    tci = results.root.trigger_clock_info
    tbl = tci.read()

    # these are timestamps from the host's (main brain's) clock

    meas_err = -tbl["start_timestamp"] + tbl["stop_timestamp"]
    if debug:
        print("meas_err.max() msec", meas_err.max() * 1e3)
        print("meas_err.min() msec", meas_err.min() * 1e3)

    # cond = meas_err < 3e-3 # take data with only small measurement errors
    cond = meas_err > -1e100  # take all data (expect measurement errors to be positive)

    # approximate timestamp (assume symmetric delays) at which clock was sampled
    mb_timestamp = (tbl["start_timestamp"][cond] + tbl["stop_timestamp"][cond]) / 2.0

    # get framenumber + fraction of next frame at which mb_timestamp estimated to happen
    framenumber = tbl["framecount"][cond]
    frac = tbl["tcnt"][cond] / float(timer_max)

    # create floating point number with this information
    framestamp = framenumber + frac

    # fit linear model of relationship mainbrain timestamp and usb trigger_device framestamp
    if len(framestamp) == 0 or len(mb_timestamp) == 0:
        raise NoTimestampDataError()
    gain, offset = model_remote_to_local(framestamp, mb_timestamp, debug=debug)
    time_model = TimeModel(gain, offset)

    dt = time_model.framestamp2timestamp(1) - time_model.framestamp2timestamp(0)
    fps_estimated = 1.0 / dt

    # Check that fps seems reasonable
    fps_saved = get_fps(results, fail_on_error=False)
    if fps_saved is not None:
        if debug:
            print("fps_estimated,fps_saved", fps_estimated, fps_saved)
            print("fps estimated from time model agrees with fps saved")
        if not np.allclose(fps_estimated, fps_saved, rtol=1e-3):
            warnings.warn(
                "fps estimated and saved are different: %s vs %s"
                % (fps_estimated, fps_saved)
            )

    if full_output:
        full_results = {
            "framestamp": framestamp,  # frame stamp on USB device
            "mb_timestamp": mb_timestamp,  # timestamp on main brain
            "gain": gain,
            "offset": offset,
        }
        return time_model, full_results
    else:
        return time_model


def drift_estimates(results):
    """calculate clock information"""
    table = results.root.host_clock_info
    remote_hostnames = np.asarray(table.read(field="remote_hostname"))
    hostnames = [x.decode() for x in list(set(remote_hostnames))]
    hostnames.sort()

    del remote_hostnames

    result = {}

    for hostname in hostnames:
        row_idx = table.get_where_list(
            "remote_hostname == hostname", condvars={"hostname": hostname}
        )
        assert len(row_idx) > 0
        start_timestamp = np.asarray(
            table.read_coordinates(row_idx, field="start_timestamp")
        )
        stop_timestamp = np.asarray(
            table.read_coordinates(row_idx, field="stop_timestamp")
        )
        remote_timestamp = np.asarray(
            table.read_coordinates(row_idx, field="remote_timestamp")
        )

        measurement_error = stop_timestamp - start_timestamp
        clock_diff = stop_timestamp - remote_timestamp

        # local time when we think remote timestamp was gathered, given symmetric transmission delays
        local_timestamp = start_timestamp + measurement_error * 0.5

        short_hostname = (
            hostname.strip()
        )  # deal with old pytables bug resulting in corrupt files

        result.setdefault("hostnames", []).append(short_hostname)
        result.setdefault("local_timestamp", {})[short_hostname] = local_timestamp
        result.setdefault("remote_timestamp", {})[short_hostname] = remote_timestamp
        result.setdefault("measurement_error", {})[short_hostname] = measurement_error
    return result


def make_exact_movie_info2(results, movie_dir=None):
    class ExactMovieInfo(PT.IsDescription):
        cam_id = PT.StringCol(16, pos=0)
        filename = PT.StringCol(255, pos=1)
        start_frame = PT.UInt64Col(pos=2)
        stop_frame = PT.UInt64Col(pos=3)
        start_timestamp = PT.FloatCol(pos=4)
        stop_timestamp = PT.FloatCol(pos=5)

    status("making exact movie info")

    movie_info = results.root.movie_info
    data2d = results.root.data2d_distorted
    cam_info = results.root.cam_info

    camn2cam_id = {}
    for row in cam_info:
        cam_id, camn = row["cam_id"], row["camn"]
        camn2cam_id[camn] = cam_id

    exact_movie_info = results.create_table(
        results.root, "exact_movie_info", ExactMovieInfo, ""
    )

    for row in movie_info:
        cam_id = row["cam_id"]
        filename = row["filename"]
        print("filename1:", filename)
        if movie_dir is None:
            computer_name = cam_id.split("_")[0]
            filename = filename.replace("~/", "~/%s/" % computer_name)
        else:
            filename = os.path.join(movie_dir, os.path.split(filename)[-1])

        print("filename2:", filename)
        filename = os.path.expanduser(filename)
        print("filename3:", filename)
        import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat

        frame_server = FlyMovieFormat.FlyMovie(filename, check_integrity=True)
        status(" for %s %s:" % (cam_id, filename))
        tmp_frame, timestamp_movie_start = frame_server.get_frame(0)
        tmp_frame, timestamp_movie_stop = frame_server.get_frame(-1)
        status("  %s %s" % (repr(timestamp_movie_start), repr(timestamp_movie_stop)))
        camn_start_frame_list = [
            (x["camn"], x["frame"])
            for x in data2d
            if x["timestamp"] == timestamp_movie_start
        ]
        ##        camn_start_frame_list = [(x['camn'],x['frame']) for x in data2d.where(
        ##            data2d.cols.timestamp == timestamp_movie_start )]
        if len(camn_start_frame_list) == 0:
            status(
                "WARNING: movie for %s %s : start data not found" % (cam_id, filename)
            )
            # ts = np.array(data2d.cols.timestamp)
            # print 'min(ts),timestamp_movie_start,max(ts)',min(ts),timestamp_movie_start,max(ts)
            continue
        else:
            if len(camn_start_frame_list) > 1:
                for camn, start_frame in camn_start_frame_list:
                    if camn2cam_id[camn] == cam_id:
                        break
            else:
                camn, start_frame = camn_start_frame_list[0]
            assert camn2cam_id[camn] == cam_id
        camn_stop_frame_list = [
            x["frame"] for x in data2d if x["timestamp"] == timestamp_movie_stop
        ]
        ##        camn_stop_frame_list = [x['frame'] for x in data2d.where(
        ##            data2d.cols.timestamp == timestamp_movie_stop )]
        if len(camn_stop_frame_list) == 0:
            status(
                "WARNING: movie for %s %s : stop data not found in data2d, using last data2d as stop point"
                % (cam_id, filename)
            )
            camn_frame_list = [
                x["frame"] for x in data2d if x["timestamp"] >= timestamp_movie_start
            ]
            stop_frame = max(camn_frame_list)
        else:
            stop_frame = camn_stop_frame_list[0]

        exact_movie_info.row["cam_id"] = cam_id
        exact_movie_info.row["filename"] = filename
        exact_movie_info.row["start_frame"] = start_frame
        exact_movie_info.row["stop_frame"] = stop_frame
        exact_movie_info.row["start_timestamp"] = timestamp_movie_start
        exact_movie_info.row["stop_timestamp"] = timestamp_movie_stop
        exact_movie_info.row.append()
        exact_movie_info.flush()


class Quick1DIndexer:
    """maintain a sorted cache of a particular 1D array to speed searches"""

    def __init__(self, frames):
        frames = np.asarray(frames)
        self.sorted_frame_idxs = np.argsort(frames)
        self.sorted_frames = frames[self.sorted_frame_idxs]
        diff = self.sorted_frames[1:] - self.sorted_frames[:-1]
        assert np.all(diff >= 0)
        nzdiff = diff[diff > 0]
        self.mindiff = np.min(nzdiff)

    def get_idxs(self, frameno):
        sorted_idx_low = self.sorted_frames.searchsorted(frameno)
        sorted_idx_high = self.sorted_frames.searchsorted(frameno + self.mindiff)
        idx = self.sorted_frame_idxs[sorted_idx_low:sorted_idx_high]
        return idx


class QuickFrameIndexer(Quick1DIndexer):
    def get_frame_idxs(self, frameno):
        return self.get_idxs(frameno)


@pytest.mark.parametrize("dtype", ["int", "float"])
def test_qi(dtype):
    frameso = [0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 6, 7, 8]
    if dtype == "int":
        frames = frameso
    elif dtype == "float":
        frames = list(map(float, frameso))
        frames.extend([0.1, 3.3, 3.04, 3.0004])
    qi = Quick1DIndexer(frames)

    for fno in np.unique(frames):
        idxs = qi.get_idxs(fno)

        idxs = list(idxs)
        while len(idxs):
            idx = idxs.pop()
            assert frames[idx] == fno

    fno = np.max(frames) + 1
    idxs = qi.get_idxs(fno)
    assert len(idxs) == 0


def do_json_progress(percent):
    class PrettyFloat(float):
        def __repr__(self):
            return "%.1f" % self

    print(json.dumps({"progress": {"percent": PrettyFloat(percent)}}))
