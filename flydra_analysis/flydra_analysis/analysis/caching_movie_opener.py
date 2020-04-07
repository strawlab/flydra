from __future__ import print_function
from __future__ import absolute_import
from . import result_utils
from . import smdfile
import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat
import numpy as nx


class NoFrameRecordedHere(Exception):
    pass


class CachingMovieOpener:
    """

    last used 2006-05-17

    """

    def __init__(self):
        self.cache = {}
        self.fmfs_by_filename = {}
        self.smds_by_fmf_filename = {}
        self.bg_image_cache = {}
        self.undistorted_bg_image_cache = {}
        self.movie_dir = None

    def set_movie_dir(self, movie_dir):
        self.movie_dir = movie_dir

    def _load_fmf_and_smd(self, indexes):
        results, camn, cam_id, frame_type, remote_timestamp = indexes
        if frame_type == "full_frame_fmf":
            if not hasattr(results.root, "exact_movie_info"):
                result_utils.make_exact_movie_info2(results, movie_dir=self.movie_dir)
            exact_movie_info = results.root.exact_movie_info

            # find normal (non background movie filename)
            found = False
            for row in exact_movie_info.where(exact_movie_info.cols.cam_id == cam_id):
                if row["start_timestamp"] < remote_timestamp < row["stop_timestamp"]:
                    filename = row["filename"]
                    found = True
                    break
            if not found:
                raise ValueError("movie not found for %s" % (cam_id,))

            filename = os.path.splitext(filename)[0] + "%s.fmf" % (
                suffix,
            )  # alter to be background image
        else:
            if hasattr(results.root, "small_fmf_summary"):
                found = False
                small_fmf_summary = results.root.small_fmf_summary
                for row in small_fmf_summary.where(small_fmf_summary.cols.camn == camn):
                    if (
                        row["start_timestamp"]
                        <= remote_timestamp
                        <= row["stop_timestamp"]
                    ):
                        found = True
                        basename = row["basename"]
                        filename = basename + ".fmf"
                        break

            elif hasattr(results.root, "exact_roi_movie_info"):
                found = False
                exact_roi_movie_info = results.root.exact_roi_movie_info
                for row in exact_roi_movie_info.where(
                    exact_roi_movie_info.cols.timestamp == remote_timestamp
                ):
                    if row["cam_id"] == cam_id:
                        filename = row["filename"]
                        found = True
                    break
            else:
                raise RuntimeError(
                    'need "small_fmf_summary" or "exact_roi_movie_info" table'
                )

            if not found:
                raise NoFrameRecordedHere(
                    "frame not found for %s, %s" % (cam_id, repr(remote_timestamp))
                )

        if filename not in self.fmfs_by_filename:
            self.fmfs_by_filename[filename] = FlyMovieFormat.FlyMovie(filename)
        if filename not in self.smds_by_fmf_filename:
            self.smds_by_fmf_filename[filename] = smdfile.SMDFile(
                filename[:-4] + ".smd"
            )
        fmf = self.fmfs_by_filename[filename]
        smd = self.smds_by_fmf_filename[filename]
        return (fmf, smd)

    def get_movie_frame(
        self,
        results,
        remote_timestamp_or_frame,
        cam,
        return_bg_instead_of_error_if_frame_missing=False,
        suffix=None,
        frame_type=None,
        width=None,
        height=None,
    ):
        if frame_type is None:
            frame_type = "full_frame_fmf"

        if frame_type not in [
            "small_frame_and_bg",
            "small_frame_only",
            "full_frame_fmf",
        ]:
            raise ValueError("unsupported frame_type")

        print("XY0")
        if frame_type != "full_frame_fmf" and suffix is not None:
            raise ValueError(
                "suffix has no meaning unless frame_type is full_frame_fmf"
            )

        if suffix is None:
            suffix = ""

        camn = None
        frame = None
        if isinstance(remote_timestamp_or_frame, float):
            remote_timestamp = remote_timestamp_or_frame
        else:
            frame = remote_timestamp_or_frame
            print("XY0.5 frame", frame)
            camn, remote_timestamp = result_utils.get_camn_and_remote_timestamp(
                results, cam, frame
            )
            print("XY0.6 remote_timestamp", remote_timestamp)

        cam_info = results.root.cam_info
        print("XY1")
        if isinstance(cam, int):  # camn
            camn = cam
            cam_id = [x["cam_id"] for x in cam_info.where(cam_info.cols.camn == camn)][
                0
            ]
            cam_id = [x["cam_id"] for x in cam_info if x["camn"] == camn][0]
        elif isinstance(cam, str):  # cam_id
            cam_id = cam

        if frame is None or camn is None:
            camn, frame = result_utils.get_camn_and_frame(
                results, cam_id, remote_timestamp
            )
        print("XY3")
        print("camn", camn)
        print("frame", frame)
        print("remote_timestamp", remote_timestamp)
        print()

        indexes = (results, camn, cam_id, frame_type, remote_timestamp)

        print("XY4")
        if indexes not in self.cache:
            print("XY4.5")
            try:
                print("XY4.6")
                self.cache[indexes] = self._load_fmf_and_smd(indexes)
                print("XY4.7")
            except NoFrameRecordedHere:
                print("XY4.8")
                self.cache[indexes] = None
                print("XY4.85")
        print("XY4.9")
        (fmf, smd) = self.cache[indexes]
        print("XY5")

        if fmf is None:
            raise NoFrameRecordedHere("")
        print("XY6")

        if frame_type == "small_frame_and_bg":
            # get background frame
            camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
            cam_id = camn2cam_id[camn]
            bg_frame = nx.asarray(results.root.backgrounds.__getattr__(cam_id))
        elif frame_type == "small_frame_only":
            if height is None or width is None:
                raise ValueError(
                    'for frame_type="small_frame_only" must specify width and height'
                )
            bg_frame = 255 * nx.ones((height, width), nx.UInt8)

        print("XY7")
        # get frame
        try:
            frame, movie_timestamp = fmf.get_frame_at_timestamp(remote_timestamp)
        except ValueError as err:
            if str(err).startswith("no frame at timestamp given"):
                if return_bg_instead_of_error_if_frame_missing:
                    frame = None
                    movie_timestamp = remote_timestamp
                else:
                    raise NoFrameRecordedHere("")
            else:
                raise

        print("XY7.1 fmf.filename", fmf.filename)
        print("XY7.2 movie_timestamp", movie_timestamp)

        # make full frame
        if frame_type != "full_frame_fmf":
            small_frame = frame
            del frame
            if small_frame is not None:
                # normal code path
                height, width = small_frame.shape

                left, bottom = smd.get_left_bottom(movie_timestamp)
                frame = bg_frame.copy()
                frame[bottom : bottom + height, left : left + width] = small_frame
            else:
                # no frame found
                frame = bg_frame
        return frame, movie_timestamp

    def get_background_image(self, results, cam_id):
        idx = (results, cam_id)
        if idx not in self.bg_image_cache:
            self.bg_image_cache[idx] = nx.asarray(
                results.root.backgrounds.__getattr__(cam_id)
            )
        bg = self.bg_image_cache[idx]
        return bg

    def _undist(self, idx, im):
        (results, reconstructor, cam_id) = idx
        intrin = reconstructor.get_intrinsic_linear(cam_id)
        k = reconstructor.get_intrinsic_nonlinear(cam_id)
        f = intrin[0, 0], intrin[1, 1]  # focal length
        c = intrin[0, 2], intrin[1, 2]  # camera center
        undist_im = undistort.rect(im, f=f, c=c, k=k)
        return undist_im

    def get_undistorted_background_image(self, results, reconstructor, cam_id):
        im = self.get_background_image(results, cam_id)
        idx = (results, reconstructor, cam_id)
        if idx not in self.undistorted_bg_image_cache:
            self.undistorted_bg_image_cache[idx] = self._undist(idx, im)
        undist = self.undistorted_bg_image_cache[idx]
        return undist
