from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import math
import numpy
from tvtk.api import tvtk
from . import conditions
from . import stimulus_positions
import flydra_core.geom as geom


def draw_cubic_solid(origin, size):
    actors = []
    x = numpy.array([1, 0, 0])
    y = numpy.array([0, 1, 0])
    z = numpy.array([0, 0, 1])

    verts = [
        origin + y * size,
        origin + y * size + x * size,
        origin + y * size + x * size + z * size,
        origin + y * size + z * size,
        origin,
        origin + x * size,
        origin + x * size + z * size,
        origin + z * size,
    ]

    # indexes into verts
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [4, 0],
        [5, 1],
        [6, 2],
        [7, 3],
    ]

    pd = tvtk.PolyData()
    pd.points = verts
    pd.lines = edges
    if 1:
        pt = tvtk.TubeFilter(
            radius=0.0254 * 0.5,
            input=pd,
            number_of_sides=12,
            vary_radius="vary_radius_off",
        )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0.9, 0.9, 0.9
        a.property.specular = 0.3
        actors.append(a)
    return actors


def simple_tvtk(verts, polys):
    actors = []

    pd = tvtk.PolyData()
    verts = numpy.array(verts)
    if verts.shape[1] == 4:
        onetest = verts[:, 3]
        assert numpy.allclose(1, onetest)
        verts = verts[:, :3]
    pd.points = verts
    pd.lines = polys
    if 1:
        pt = tvtk.TubeFilter(
            radius=0.001, input=pd, number_of_sides=12, vary_radius="vary_radius_off",
        )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0.9, 0.9, 0.9
        a.property.specular = 0.3
        actors.append(a)
    return actors


class DrawBase(object):
    pass


class hum07(DrawBase):
    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        self.kwargs = kwargs

    def get_tvtk_actors(self):
        filename = self.filename
        kwargs = self.kwargs

        actors = []
        # humdra setup 20071130

        ft2inch = 12
        inch2cm = 2.54
        cm2m = 0.01
        ft2m = ft2inch * inch2cm * cm2m
        origin = numpy.array([0, 0, 0])
        size = numpy.array([5 * ft2m, 10 * ft2m, 5 * ft2m])
        actors.extend(draw_cubic_solid(origin, size))
        # actors.extend(draw_cubic_solid(origin,0.1*size))

        return actors


class mama07(DrawBase):
    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        self.kwargs = kwargs

    def get_tvtk_actors(self):
        filename = self.filename
        kwargs = self.kwargs

        actors = []

        # mamarama
        z = 0
        N = 64
        radius = 1.0
        center = numpy.array([1.0, 0, 0])

        verts = []
        vi = 0  # vert idx
        lines = []
        theta = numpy.linspace(0, 2 * math.pi, N, endpoint=False)
        X = radius * numpy.cos(theta) + center[0]
        Y = radius * numpy.sin(theta) + center[1]
        height = 0.762
        for z in numpy.linspace(0, height, 4):
            Z = numpy.zeros(theta.shape) + center[2] + z
            v = numpy.array([X, Y, Z]).T
            for i in range(N):
                verts.append(v[i])

            for i in range(N - 1):
                lines.append([i + vi, i + 1 + vi])
            lines.append([vi + N - 1, vi])

            vi += N

        pd = tvtk.PolyData()
        pd.points = verts
        pd.lines = lines
        pt = tvtk.TubeFilter(
            radius=0.001, input=pd, number_of_sides=4, vary_radius="vary_radius_off",
        )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0.9, 0.9, 0.9
        a.property.specular = 0.3
        actors.append(a)
        return actors


class mama20080414(DrawBase):
    def __init__(self, filename=None, **kwargs):
        self.kwargs = kwargs
        self.z = 0
        self.N = 64
        self.radius = 1.0
        self.center = numpy.array([0.1, 0.8, 0])
        self.height = 0.762

    def _get_verts_lines(self):
        # mamarama
        z = self.z
        N = self.N
        radius = self.radius
        center = self.center
        height = self.height

        verts = []
        vi = 0  # vert idx
        lines = []
        theta = numpy.linspace(0, 2 * math.pi, N, endpoint=False)
        X = radius * numpy.cos(theta) + center[0]
        Y = radius * numpy.sin(theta) + center[1]
        for z in numpy.linspace(0, height, 4):
            Z = numpy.zeros(theta.shape) + center[2] + z
            v = numpy.array([X, Y, Z]).T
            for i in range(N):
                verts.append(v[i])

            for i in range(N - 1):
                lines.append([i + vi, i + 1 + vi])
            lines.append([vi + N - 1, vi])

            vi += N
        return verts, lines

    def get_3d_lines(self):
        result = []

        verts, lines = self._get_verts_lines()
        for line in lines:
            this_line = numpy.array([verts[i] for i in line])
            result.append(this_line)
        return result

    def get_tvtk_actors(self):
        actors = []

        verts, lines = self._get_verts_lines()

        pd = tvtk.PolyData()
        pd.points = verts
        pd.lines = lines
        pt = tvtk.TubeFilter(
            radius=0.001, input=pd, number_of_sides=4, vary_radius="vary_radius_off",
        )
        m = tvtk.PolyDataMapper(input=pt.output)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0.9, 0.9, 0.9
        a.property.specular = 0.3
        actors.append(a)
        return actors


class mama20080501(mama20080414):
    def __init__(self, filename=None, **kwargs):
        self.kwargs = kwargs
        self.z = 0
        self.N = 64
        self.radius = 1.0
        self.center = numpy.array([0.1, 0.5, 0])
        self.height = 0.762


class default(DrawBase):
    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        self.kwargs = kwargs

    def get_tvtk_actors(self):
        filename = self.filename
        kwargs = self.kwargs

        ### draw floor
        actors = []

        x0 = -1.5 / 2
        x1 = 1.5 / 2
        y0 = -0.305 / 2
        y1 = 0.305 / 2
        z0 = 0.314

        inc = 0.05
        if 1:
            nx = int(math.ceil((x1 - x0) / inc))
            ny = int(math.ceil((y1 - y0) / inc))
            eps = 1e-10
            x1 = x0 + nx * inc + eps
            y1 = y0 + ny * inc + eps

        segs = []
        for x in numpy.r_[x0:x1:inc]:
            seg = [(x, y0, z0), (x, y1, z0)]
            segs.append(seg)
        for y in numpy.r_[y0:y1:inc]:
            seg = [(x0, y, z0), (x1, y, z0)]
            segs.append(seg)

        if 1:
            verts = []
            for seg in segs:
                verts.extend(seg)
            verts = numpy.asarray(verts)

            pd = tvtk.PolyData()

            np = len(verts) / 2
            lines = numpy.zeros((np, 2), numpy.int64)
            lines[:, 0] = 2 * numpy.arange(np, dtype=numpy.int64)
            lines[:, 1] = lines[:, 0] + 1

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

        ###################
        stim = None
        try:
            condition, stim = conditions.get_condition_stimname_from_filename(
                filename, **kwargs
            )
            print('Data from condition "%s",with stimulus' % (condition,), stim)
        except KeyError as err:
            if kwargs.get("force_stimulus", False):
                raise
            else:
                print("Unknown condition and stimname")

        if stim is None:
            return actors

        all_verts = stimulus_positions.stim_positions[stim]

        for verts in all_verts:

            verts = numpy.asarray(verts)
            pd = tvtk.PolyData()

            np = len(verts) - 1
            lines = numpy.zeros((np, 2), numpy.int64)
            lines[:, 0] = numpy.arange(0, np - 0.5, 1, numpy.int64)
            lines[:, 1] = numpy.arange(1, np + 0.5, 1, numpy.int64)

            pd.points = verts
            pd.lines = lines

            pt = tvtk.TubeFilter(
                radius=0.006,
                input=pd,
                number_of_sides=20,
                vary_radius="vary_radius_off",
            )
            m = tvtk.PolyDataMapper(input=pt.output)
            a = tvtk.Actor(mapper=m)
            a.property.color = 0, 0, 0
            a.property.specular = 0.3
            actors.append(a)
        return actors


class wt0803(DrawBase):
    def __init__(self, **kwargs):
        pass

    def get_lines(self):
        x1 = (0.61809734343982814, -0.013406307631868021, -0.0035917469609576813)
        x2 = (1.198826951196984, -0.012485555989418704, 0.0025602971809217849)
        line_mark = geom.line_from_points(geom.ThreeTuple(x1), geom.ThreeTuple(x2))
        # line_mark = geom.line_from_HZline( (0.02439605071482559, 0.051420313894467287, -0.99564385219908413, -0.00097158319016768459, 0.015447764310623923, -0.072211676414489459))
        # (-0.061248834007043639, -0.25331384274265706, 0.92406568281060819, 0.0036304349314720606, -0.053126804348089779, 0.27449527110946625))

        # (0.0097110036703961945, 0.071801811107512326, -0.98489513490720026, 0.00048467528993275467, 0.027587746154489213, -0.15482393522159865))
        # (-0.072175447568079532, -0.31684664370365356, 0.86817678663756903, 0.0048959408494628444, -0.070490367875258558, 0.36834102959859433))
        # (0.026432522291979302, 0.066719950163430089, -0.98930078556948686, -0.0017052044659353833, 0.024116927663042458, -0.12469639836376363))
        # (0.036051887242803483, 0.1225934342601759, -0.97241823143838102, -0.0025815311891907609, 0.035919256583691377, -0.19177340058510137))
        line_clean = geom.line_from_HZline(
            (
                -0.27936512712372191,
                0.04624199569732089,
                -0.9586296181568541,
                -0.0080073601565274575,
                0.0023435304010169695,
                0.027864831277130742,
            )
        )

        x1 = (0.49949904826195857, -0.012373271634116311, 0.32286252608243299)
        x2 = (0.38901860101731867, -0.01294145950243236, 0.32642509461777808)
        top_mark_line = geom.line_from_points(geom.ThreeTuple(x1), geom.ThreeTuple(x2))
        # top_mark_line= geom.line_from_HZline((0.18050156903291037, 0.31868846699973336, -0.73306806601973318, -0.065965871714262242, 0.15803989259450685, -0.54693688052443501))
        # (0.17074976404913067, 0.32855038182801033, -0.67404658417915053, -0.072283972587872641, 0.16975362231152435, -0.61197965820220468))
        # (0.17544381933023984, 0.49024035729704918, -0.56597775796347805, -0.05688098204074131, 0.1552881996760179, -0.61741652451985551))
        top_clean_line = geom.line_from_HZline(
            (
                -0.26575764351965903,
                -0.29763801029854864,
                -0.91692940763466335,
                -0.0018355385669680962,
                0.0018496262007779841,
                0.0042615527956441791,
            )
        )

        verts = []
        lines = []
        count = 0
        for line in [line_mark, line_clean, top_mark_line, top_clean_line]:

            this_verts = []
            this_lines = []

            d = line.u
            direction = numpy.array([d[0], d[1], d[2]])
            norm = direction / numpy.sqrt(numpy.sum(direction ** 2))
            d = line.closest()
            pt0 = numpy.array([d[0], d[1], d[2]])

            for d in numpy.linspace(-2, 2, 100):
                this_verts.append(pt0 + d * norm)

            for i in range(len(this_verts) - 1):
                this_lines.append([count + i, count + i + 1])

            count += len(this_verts)

            ##             print 'this_verts[0]',this_verts[0]
            ##             print 'this_verts[-1]',this_verts[-1]
            ##             print

            verts.extend(this_verts)
            lines.extend(this_lines)

        # make homogeneous
        verts = [numpy.array((v[0], v[1], v[2], 1.0)) for v in verts]
        ##         print 'verts\n',
        ##         for v in verts:
        ##             print v
        ##         print 'lines\n',
        ##         for l in lines:
        ##             print l

        return verts, lines

    def get_lines_broken(self):
        if 1:
            mark_downwind = (
                -0.34217471182253395,
                -0.049713741423482401,
                -0.19355194735726916,
            )
            mark_upwind = (
                1.2876590202076592,
                0.010489177564320891,
                0.12787225786089762,
            )
            clean_downwind = (
                -0.056335257988488305,
                0.29128361824544396,
                -0.04660008661573465,
            )
            clean_upwind = (
                1.6976573714828826,
                0.29557154652605472,
                -0.097584020093109847,
            )

            topmark_downwind = (
                0.53170860804444731,
                -0.16409787403831047,
                -0.28615024916121734,
            )  # (1.3096717388755901, 0.091305659429098823, 0.732811634933068)
            topmark_upwind = (
                0.9631957795119126,
                -0.0033269269379037625,
                0.35236001377287235,
            )
            topclean_downwind = (
                0.45463209212301769,
                0.29075143705630435,
                0.3224900075984799,
            )
            topclean_upwind = (
                1.0078270653666062,
                0.29186733966356126,
                0.3199189595274578,
            )

        else:
            mark_downwind = (
                -0.33526447401078729,
                -0.048931719337347232,
                -0.189574764923057,
            )
            mark_upwind = (
                1.3328671859305408,
                0.0064117386996079481,
                0.10473711070863806,
            )
            clean_downwind = (
                0.1647028850715552,
                0.27253360129016713,
                0.072188689328985672,
            )
            clean_upwind = (
                1.9200691965703456,
                0.31092100990246491,
                -0.2212166344677361,
            )

        verts = [
            mark_downwind,  # 0
            mark_upwind,  # 1
            clean_upwind,  # 2
            clean_downwind,  # 3
            topmark_downwind,  # 4
            topmark_upwind,  # 5
            topclean_downwind,  # 6
            topclean_upwind,  # 7
        ]

        # make homogeneous coords
        verts = [numpy.array((v[0], v[1], v[2], 1.0)) for v in verts]

        if 0:
            lines = [
                [0, 1, 2, 3],  # floor
                [0, 1, 5, 4],  # mark wall
                [2, 3, 6, 7],  # clean wall
            ]
        else:
            lines = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # floor
                [1, 5],
                [5, 4],
                [4, 0],  # mark wall
                [3, 6],
                [6, 7],
                [7, 2],  # clean wall
            ]

        return verts, lines

    def get_tvtk_actors(self):
        verts, lines = self.get_lines()
        return simple_tvtk(verts, lines)
