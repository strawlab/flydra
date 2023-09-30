from __future__ import print_function
from __future__ import absolute_import
import xml.etree.ElementTree as ET
import flydra_core.reconstruct as reconstruct
import numpy
import numpy as np
import os
import hashlib
from .core_analysis import parse_seq


class WrongXMLTypeError(Exception):
    pass


class StimProjection(object):
    """Project 3D world coordinates to 2D coordinates for plotting.

    This is an abstract base class that should be subclassed."""

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)

    def project(self, X):
        raise NotImplementedError("abstract base class method called")


class FlydraReconstructProjection(StimProjection):
    """Project coordinates using flydra camera calibration.

    Subclass of :class:`StimProjection`.
    """

    def __init__(self, R, cam_id):
        self.R = R
        self.w, self.h = R.get_resolution(cam_id)
        self.cam_id = cam_id

    def project(self, X):
        x, y = self.R.find2d(self.cam_id, X, distorted=True)
        if (x < 0) or (x > (self.w - 1)):
            return None
        if (y < 0) or (y > (self.h - 1)):
            return None
        return x, y


class SimpleOrthographicXYProjection(StimProjection):
    """Project coordinates to an XY (top) view.

    Subclass of :class:`StimProjection`.
    """

    def project(self, X):
        return X[0], X[1]


class SimpleOrthographicXZProjection(StimProjection):
    """Project coordinates to an XZ (side) view.

    Subclass of :class:`StimProjection`.
    """

    def project(self, X):
        return X[0], X[2]


class Stimulus(object):
    '''Stimulus information saved in an XML node

    Parameters
    ----------
    root : :mod:`xml.etree.ElementTree` node
        An XML node specifying the root of the stimulus tree.
        Must have tag ``stimxml``.


    Examples
    --------

    The following example specifies a stimulus-only .xml file. It
    defines a rectilinear arena aligned with the X, Y and Z axes of
    the world coordinate systems. The floor of the area is at z=0, the
    ceiling is at z=0.3 and so on.

    >>> stimulus_file_contents = """<stimxml version="1">
    ...   <cubic_arena>
    ...     <verts4x4>
    ...       <vert>.5,  .15,  .3</vert>
    ...       <vert>.5,  -.15, .3</vert>
    ...       <vert>-.5, -.15, .3</vert>
    ...       <vert>-.5,  .15, .3</vert>
    ...
    ...       <vert>.5,  .15,  0</vert>
    ...       <vert>.5,  -.15, 0</vert>
    ...       <vert>-.5, -.15, 0</vert>
    ...       <vert>-.5,  .15, 0</vert>
    ...     </verts4x4>
    ...     <tube_diameter>0.002</tube_diameter>
    ...   </cubic_arena>
    ... </stimxml>"""
    >>> root=ET.fromstring(stimulus_file_contents)
    >>> stim=Stimulus(root)
    >>> #import matplotlib
    >>> #matplotlib.use("Agg") # for running without X display
    >>> #import matplotlib.pyplot as pyplot
    >>> #ax=pyplot.subplot(1,1,1)
    >>> #stim.plot_stim(ax,SimpleOrthographicXYProjection())
    >>> #pyplot.show()
    '''

    def __init__(self, root):
        assert root.tag == "stimxml"
        assert root.attrib["version"] == "1"
        self.root = root
        self._R = None

    def get_root(self):
        """get the root XML node

        Returns
        -------
        node
            Root XML node
        """
        return self.root

    def _get_reconstructor(self):
        if self._R is None:
            r_node = self.root.find("multi_camera_reconstructor")
            self._R = reconstruct.Reconstructor_from_xml(r_node)
        return self._R

    def has_reconstructor(self):
        """Check for presence of reconstructor

        Returns
        -------
        boolean
            Whether a reconstructor exists
        """

        try:
            R = self._get_reconstructor()
        except:
            return False
        else:
            return True

    def verify_reconstructor(self, other_R):
        """verify that the reconstructor in the XML node is equal to other_R

        Parameters
        ----------
        other_R : :class:`flydra_core.reconstruct.Reconstructor` instance.
            The reconstructor to compare against.

        This function raises an exception if the reconstructors are not equal.

        Check for presence of reconstructor with :method:`has_reconstructor`.
        """
        R = self._get_reconstructor()
        assert isinstance(other_R, reconstruct.Reconstructor)
        assert R == other_R

    def verify_timestamp(self, timestamp):
        """deprecated function"""
        if 1:
            import warnings

            warnings.warn(
                "This function does not do anything anymore.",
                DeprecationWarning,
                stacklevel=2,
            )
            return

    def _get_info_for_cylindrical_arena(self, child):
        assert child.tag == "cylindrical_arena"
        info = {}
        for v in child:
            if v.tag == "origin":
                info["origin"] = numpy.fromstring(v.text, sep=" ")
            elif v.tag == "axis":
                info["axis"] = numpy.fromstring(v.text, sep=" ")
            elif v.tag == "diameter":
                info["diameter"] = float(v.text)
            elif v.tag == "height":
                info["height"] = float(v.text)
            else:
                raise ValueError("unknown tag: %s" % v.tag)
        return info

    def _get_info_for_sphere_arena(self, child):
        assert child.tag == "sphere_arena"
        info = {}
        for v in child:
            if v.tag == "origin":
                info["origin"] = numpy.fromstring(v.text, sep=" ")
            elif v.tag == "radius":
                info["radius"] = float(v.text)
            else:
                raise ValueError("unknown tag: %s" % v.tag)
        return info

    def _get_info_for_cubic_arena(self, child):
        assert child.tag == "cubic_arena"
        info = {}
        did_it = False
        for child1 in child:
            if child1.tag == "verts4x4":
                if did_it:
                    raise ValueError(
                        "already parsed a verts4x4 attribute " "for this cubic_arena"
                    )
                did_it = True
                verts = []
                for v in child1:
                    if v.tag == "vert":
                        vtext = v.text.replace(",", " ")
                        verts.append(numpy.fromstring(vtext, sep=" "))
                        assert len(verts[-1]) == 3
                assert len(verts) == 8
                info["verts4x4"] = verts
            elif child1.tag == "tube_diameter":
                info["tube_diameter"] = float(child1.text)
            else:
                raise ValueError("unknown tag: %s" % child1.tag)
        return info

    def _get_info_for_cylindrical_post(self, child):
        assert child.tag == "cylindrical_post"
        verts = []
        for v in child:
            if v.tag == "vert":
                vtext = v.text.replace(",", " ")
                verts.append(numpy.fromstring(vtext, sep=" "))
            elif v.tag == "diameter":
                diameter = float(v.text)
            else:
                raise ValueError("unknown tag: %s" % v.tag)
        return {"verts": verts, "diameter": diameter}

    def iterate_posts(self):
        """A generator to iterate over all ``cylindrical_post`` tags

        Returns
        -------
        results : iterator of dictionaries
            Each dictionary has 'verts' and 'diameter' keys whose values specify
            the post.
        """
        for child in self.root:
            if child.tag == "cylindrical_post":
                yield self._get_info_for_cylindrical_post(child)

    def get_tvtk_actors(self):
        """Get a list of actors to use in TVTK scene

        Returns
        -------
        actors : list of tvtk actors
            A list of actors to insert into tvtk scene.

        """
        actors = []
        import flydra_analysis.a2.experiment_layout as experiment_layout

        for child in self.root:
            if child.tag in ["multi_camera_reconstructor", "valid_h5_times"]:
                continue
            elif child.tag == "cylindrical_arena":
                info = self._get_info_for_cylindrical_arena(child)
                actors.extend(experiment_layout.cylindrical_arena(info=info))
            elif child.tag == "sphere_arena":
                info = self._get_info_for_sphere_arena(child)
                actors.extend(experiment_layout.sphere_arena(info=info))
            elif child.tag == "cubic_arena":
                info = self._get_info_for_cubic_arena(child)
                actors.extend(experiment_layout.cubic_arena(info=info))
            elif child.tag == "cylindrical_post":
                info = self._get_info_for_cylindrical_post(child)
                actors.extend(experiment_layout.cylindrical_post(info=info))
            else:
                import warnings

                warnings.warn("Unknown node: %s" % child.tag)
        return actors

    def draw_in_mayavi_scene(self, engine):
        """draw a representation of the stimulus in a Mayavi2 scene"""
        import flydra_analysis.a2.experiment_layout as experiment_layout

        for child in self.root:
            if child.tag in ["multi_camera_reconstructor", "valid_h5_times"]:
                continue
            elif child.tag == "cubic_arena":
                info = self._get_info_for_cubic_arena(child)
                experiment_layout.get_mayavi_cubic_arena_source(engine, info=info)
            else:
                import warnings

                warnings.warn("Unknown node: %s" % child.tag)

    def get_distorted_linesegs(self, cam_id):
        """Return line segments for the stimulus distorted by camera model"""
        # mainly copied from self.plot_stim_over_distorted_image()
        R = self._get_reconstructor()
        R = R.get_scaled()
        P = FlydraReconstructProjection(R, cam_id)
        return self._get_linesegs(projection=P)

    def _get_linesegs(self, projection=None):
        """good for OpenGL type stuff"""
        if not isinstance(projection, StimProjection):
            raise ValueError(
                "projection must be instance of " "xml_stimulus.StimProjection class"
            )
        plotted_anything = False

        linesegs = []
        lineseg_colors = []

        for child in self.root:
            if child.tag in ["multi_camera_reconstructor", "valid_h5_times"]:
                continue
            elif child.tag == "cylindrical_arena":
                plotted_anything = True
                info = self._get_info_for_cylindrical_arena(child)
                assert numpy.allclose(
                    info["axis"], numpy.array([0, 0, 1])
                ), "only vertical areas supported at the moment"

                N = 128
                theta = numpy.linspace(0, 2 * numpy.pi, N)
                r = info["diameter"] / 2.0
                xs = r * numpy.cos(theta) + info["origin"][0]
                ys = r * numpy.sin(theta) + info["origin"][1]

                z_levels = numpy.linspace(
                    info["origin"][2], info["origin"][2] + info["height"], 5
                )
                for z in z_levels:
                    this_lineseg = []
                    for x, y in zip(xs, ys):
                        X = x, y, z
                        result = projection(X)
                        if result is None:
                            # reset line segment
                            if len(this_lineseg) > 2:
                                linesegs.append(this_lineseg)
                                lineseg_colors.append((1, 0.5, 0.5, 1))
                                this_lineseg = []
                        else:
                            this_lineseg.extend(result)
                    if len(this_lineseg) > 2:
                        linesegs.append(this_lineseg)
                        lineseg_colors.append((1, 0.5, 0.5, 1))
            elif child.tag == "sphere_arena":
                raise NotImplementedError()
            elif child.tag == "cylindrical_post":
                plotted_anything = True
                info = self._get_info_for_cylindrical_post(child)
                xs, ys = [], []
                # XXX TODO: extrude line into cylinder
                this_lineseg = []
                Xs = info["verts"]
                assert len(Xs) == 2
                Xs = np.asarray(Xs)
                X0 = Xs[0]
                dir_vec = Xs[1] - Xs[0]
                fracs = np.linspace(0, 1, 100)
                for frac in fracs:
                    result = projection(X0 + frac * dir_vec)
                    if result is None:
                        # reset line segment
                        if len(this_lineseg) > 2:
                            linesegs.append(this_lineseg)
                            lineseg_colors.append((0.5, 0.5, 1, 1))
                            this_lineseg = []
                    else:
                        this_lineseg.extend(result)
                if len(this_lineseg) > 2:
                    linesegs.append(this_lineseg)
                    lineseg_colors.append((0.5, 0.5, 1, 1))
            else:
                import warnings

                warnings.warn("Unknown node: %s" % child.tag)
        if not plotted_anything:
            import warnings

            warnings.warn("Did not plot any stimulus")
        if 0:
            return [(-1, -1, 10, 20, 30, 40, 30, 400)], [(1, 0.5, 0.5, 1)]
        return linesegs, lineseg_colors

    def plot_stim_over_distorted_image(self, ax, cam_id):
        """use matplotlib to plot stimulus in image coordinates

        Parameters
        ----------
        ax : matplotlib Axes instance
            The axes into which the stimulus is drawn
        cam_id : string-like
            Specifies the name of the camera to draw the stimulus for
        """
        # we want to work with scaled coordinates
        R = self._get_reconstructor()
        R = R.get_scaled()
        P = FlydraReconstructProjection(R, cam_id)
        self.plot_stim(ax, projection=P)

    def plot_stim(
        self,
        ax,
        projection=None,
        post_colors=None,
        show_post_num=False,
        draw_post_as_circle=False,
    ):
        """use matplotlib to plot stimulus with arbitrary projection

        Parameters
        ----------
        ax : matplotlib Axes instance
            The axes into which the stimulus is drawn
        projection : :class:`StimProjection` instance
            Specifies the projection to plot the stimulus with
        post_colors : (optional)
            Post colors when drawing ``cylindrical_post`` tags.
        show_post_num : boolean
            Whether to show post number when drawing
            ``cylindrical_post`` tags. Defaults to False.
        draw_post_as_circle : boolean
            Whether to draw post as circle when drawing
            ``cylindrical_post`` tags. Defaults to False.

        """
        if not isinstance(projection, StimProjection):
            print("projection", projection)
            raise ValueError(
                "projection must be instance of " "xml_stimulus.StimProjection class"
            )

        plotted_anything = False

        post_num = 0
        for child in self.root:
            if child.tag in ["multi_camera_reconstructor", "valid_h5_times"]:
                continue
            elif child.tag == "cylindrical_arena":
                plotted_anything = True
                info = self._get_info_for_cylindrical_arena(child)
                assert numpy.allclose(
                    info["axis"], numpy.array([0, 0, 1])
                ), "only vertical areas supported at the moment"

                N = 128
                theta = numpy.linspace(0, 2 * numpy.pi, N)
                r = info["diameter"] / 2.0
                xs = r * numpy.cos(theta) + info["origin"][0]
                ys = r * numpy.sin(theta) + info["origin"][1]

                z_levels = numpy.linspace(
                    info["origin"][2], info["origin"][2] + info["height"], 5
                )
                for z in z_levels:
                    plotx, ploty = [], []
                    for x, y in zip(xs, ys):
                        X = x, y, z
                        result = projection(X)
                        if result is None:
                            x2d, y2d = np.nan, np.nan
                        else:
                            x2d, y2d = result
                        plotx.append(x2d)
                        ploty.append(y2d)
                    ax.plot(plotx, ploty, "k-")
            elif child.tag == "sphere_arena":
                raise NotImplementedError()
            elif child.tag == "cubic_arena":
                plotted_anything = True
                info = self._get_info_for_cubic_arena(child)
                v = info["verts4x4"]  # arranged in 2 rectangles of 4 verts
                points = [projection(v[i]) for i in range(len(v))]
                lines = [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
                for line in lines:
                    v0 = points[line[0]]
                    v1 = points[line[1]]
                    plotx = [v0[0], v1[0]]
                    ploty = [v0[1], v1[1]]
                    ax.plot(plotx, ploty, "k-")
            elif child.tag == "cylindrical_post":
                plotted_anything = True
                info = self._get_info_for_cylindrical_post(child)
                xs, ys = [], []
                assert len(info["verts"]) == 2
                r = info["diameter"] / 2.0
                X0 = info["verts"][0]
                X1 = info["verts"][1]

                if post_colors is None:
                    post_color = "k"
                else:
                    post_color = post_colors[post_num]

                if draw_post_as_circle:
                    Xav = (X0 + X1) / 2.0
                    theta = np.linspace(0, 2 * np.pi, 100)
                    xo = r * np.cos(theta)
                    yo = r * np.sin(theta)
                    ax.fill(
                        Xav[0] + xo,
                        Xav[1] + yo,
                        "-",
                        facecolor=post_color,
                        edgecolor="none",
                    )
                else:
                    dX = X1 - X0
                    for frac in np.linspace(0, 1.0, 50):
                        X = X0 + frac * dX
                        result = projection(X)
                        if result is None:
                            v2x, v2y = np.nan, np.nan
                        else:
                            v2x, v2y = result
                        xs.append(v2x)
                        ys.append(v2y)

                    ax.plot(xs, ys, "-", color=post_color, linewidth=5)

                if show_post_num:
                    ax.text(xs[0], ys[0], "post %d" % post_num)
                post_num += 1
            else:
                import warnings

                warnings.warn("Unknown node: %s" % child.tag)
        if not plotted_anything:
            import warnings

            warnings.warn("Did not plot any stimulus")


class StimulusFanout(object):
    def __init__(self, root, my_directory=None):
        if not root.tag == "stimulus_fanout_xml":
            raise WrongXMLTypeError("not the correct xml type")
        assert root.attrib["version"] == "1"
        self.root = root
        self._my_directory = my_directory

    def _get_episode_for_timestamp(self, timestamp_string=None):
        bad_md5_found = False
        for single_episode in self.root.findall("single_episode"):
            for kh5_file in single_episode.findall("kh5_file"):
                fname = kh5_file.attrib["name"]
                fname_timestamp_string = os.path.splitext(os.path.split(fname)[-1])[0][
                    4:19
                ]
                if fname_timestamp_string == timestamp_string:
                    if 1:
                        # check that the file has not changed
                        expected_md5 = kh5_file.attrib["md5sum"]
                        m = hashlib.md5()
                        if self._my_directory is not None:
                            fname = os.path.join(self._my_directory, fname)
                        m.update(open(fname, mode="rb").read())
                        actual_md5 = m.hexdigest()
                        if not expected_md5 == actual_md5:
                            bad_md5_found = True
                    stim_fname = single_episode.find("stimxml_file").attrib["name"]
                    if self._my_directory is not None:
                        stim_fname = os.path.join(self._my_directory, stim_fname)
                    if bad_md5_found:
                        raise ValueError(
                            "could not find timestamp_string '%s' "
                            "with valid md5sum. (Bad md5sum "
                            "found.)" % timestamp_string
                        )
                    # return first found episode -- XXX TODO return them all!
                    return single_episode, kh5_file, stim_fname
        raise ValueError("could not find timestamp_string '%s'" % (timestamp_string,))

    def get_walking_start_stops_for_timestamp(self, timestamp_string=None):
        (single_episode, kh5_file, stim_fname) = self._get_episode_for_timestamp(
            timestamp_string=timestamp_string
        )
        start_stops = []
        for walking in single_episode.findall("walking"):
            start = walking.attrib.get("start", None)  # frame number
            stop = walking.attrib.get("stop", None)  # frame number
            if start is not None:
                start = int(start)
            if stop is not None:
                stop = int(stop)
            start_stops.append((start, stop))
        return start_stops

    def get_obj_ids_for_timestamp(self, timestamp_string=None):
        (single_episode, kh5_file, stim_fname) = self._get_episode_for_timestamp(
            timestamp_string=timestamp_string
        )
        for include in single_episode.findall("include"):
            raise ValueError("<single_episode> has <include> tag")
        for exclude in single_episode.findall("exclude"):
            raise ValueError("<single_episode> has <exclude> tag")
        include_ids = None
        exclude_ids = None
        for include in kh5_file.findall("include"):
            if include_ids is None:
                include_ids = []
            if include.text is not None:
                obj_ids = parse_seq(include.text)
                include_ids.extend(obj_ids)
            else:
                obj_ids = None
        for exclude in kh5_file.findall("exclude"):
            if exclude_ids is None:
                exclude_ids = []
            if exclude.text is not None:
                obj_ids = parse_seq(exclude.text)
                exclude_ids.extend(obj_ids)
            else:
                obj_ids = None
        return include_ids, exclude_ids

    def get_stimulus_for_timestamp(self, timestamp_string=None):
        (single_episode, kh5_file, stim_fname) = self._get_episode_for_timestamp(
            timestamp_string=timestamp_string
        )
        root = ET.parse(stim_fname).getroot()
        stim = Stimulus(root)
        # stim.verify_timestamp( fname_timestamp_string )
        return stim

    def iterate_single_episodes(self):
        for child in self.root:
            if child.tag == "single_episode":
                yield child


def xml_fanout_from_filename(filename):
    root = ET.parse(filename).getroot()
    my_directory = os.path.split(filename)[0]
    sf = StimulusFanout(root, my_directory=my_directory)
    return sf


def xml_stimulus_from_filename(filename, timestamp_string=None):
    root = ET.parse(filename).getroot()
    if root.tag == "stimxml":
        return Stimulus(root)
    elif root.tag == "stimulus_fanout_xml":
        assert timestamp_string is not None
        sf = xml_fanout_from_filename(filename)
        stim = sf.get_stimulus_for_timestamp(timestamp_string=timestamp_string)
        return stim
    else:
        raise ValueError("unknown XML file")


def print_kh5_files_in_fanout(filename):
    sf = xml_fanout_from_filename(filename)
    for single_episode in sf.root.findall("single_episode"):
        for kh5_file in single_episode.findall("kh5_file"):
            print(kh5_file.attrib["name"], end=" ")
    print()


def main():
    import sys

    filename = sys.argv[1]
    print_kh5_files_in_fanout(filename)


def print_linesegs():
    import sys

    filename = sys.argv[1]
    stim = xml_stimulus_from_filename(filename)
    linesegs, colors = stim.get_distorted_linesegs("mama01_0")
    for i in range(len(linesegs)):
        print(linesegs[i])
        print(colors[i])
        print()


if __name__ == "__main__":
    main()
