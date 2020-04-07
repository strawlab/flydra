from __future__ import absolute_import
from . import conditions_draw
import numpy
import numpy as np
from numpy import array
import warnings

from tvtk.api import tvtk
from mayavi.sources.api import VTKDataSource
from mayavi.modules.surface import Surface
from mayavi.modules.vectors import Vectors

from tvtk.common import configure_input_data


def cylindrical_arena(info=None):
    assert numpy.allclose(
        info["axis"], numpy.array([0, 0, 1])
    ), "only vertical areas supported at the moment"

    N = 128
    theta = numpy.linspace(0, 2 * numpy.pi, N)
    r = info["diameter"] / 2.0
    xs = r * numpy.cos(theta) + info["origin"][0]
    ys = r * numpy.sin(theta) + info["origin"][1]

    z_levels = numpy.linspace(info["origin"][2], info["origin"][2] + info["height"], 5)

    verts = []
    vi = 0  # vert idx
    lines = []

    for z in z_levels:
        zs = z * numpy.ones_like(xs)
        v = numpy.array([xs, ys, zs]).T
        for i in range(N):
            verts.append(v[i])

        for i in range(N - 1):
            lines.append([i + vi, i + 1 + vi])
        lines.append([vi + N - 1, vi])

        vi += N
    pd = tvtk.PolyData(points=verts, lines=lines)
    pt = tvtk.TubeFilter(
        radius=0.001, number_of_sides=4, vary_radius="vary_radius_off",
    )
    configure_input_data(pt, pd)
    m = tvtk.PolyDataMapper(input_connection=pt.output_port)
    a = tvtk.Actor(mapper=m)
    a.property.color = 0.9, 0.9, 0.9
    a.property.specular = 0.3
    return [a]


def sphere_arena(info=None):
    N = 32
    theta = numpy.linspace(0, 2 * numpy.pi, N)
    r = info["radius"]
    xs = r * numpy.cos(theta) + info["origin"][0]
    ys = r * numpy.sin(theta) + info["origin"][1]

    els = numpy.linspace(-np.pi / 2, np.pi / 2, 10)

    # z_levels = numpy.linspace(info['origin'][2]-info['radius'],
    #                           info['origin'][2]+info['radius'],
    #                           5)

    verts = []
    vi = 0  # vert idx
    lines = []

    for el in els:
        zs = np.sin(el) * numpy.ones_like(xs) * info["radius"] + info["origin"][2]
        R = np.cos(el)
        v = numpy.array([R * xs, R * ys, zs]).T
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
    return [a]


def cylindrical_post(info=None):
    verts = info["verts"]
    diameter = info["diameter"]

    radius = diameter / 2.0
    actors = []
    verts = numpy.asarray(verts)
    pd = tvtk.PolyData()

    np = len(verts) - 1
    lines = numpy.zeros((np, 2), numpy.int64)
    lines[:, 0] = numpy.arange(0, np - 0.5, 1, numpy.int64)
    lines[:, 1] = numpy.arange(1, np + 0.5, 1, numpy.int64)

    pd.points = verts
    pd.lines = lines

    pt = tvtk.TubeFilter(
        radius=radius, input=pd, number_of_sides=20, vary_radius="vary_radius_off",
    )
    m = tvtk.PolyDataMapper(input=pt.output)
    a = tvtk.Actor(mapper=m)
    a.property.color = 0, 0, 0
    a.property.specular = 0.3
    actors.append(a)
    return actors


def cubic_arena(info=None):
    tube_radius = info["tube_diameter"] / 2.0

    def make_2_vert_tube(a, b):
        pd = tvtk.PolyData(points=[a, b], lines=[[0, 1]])
        pt = tvtk.TubeFilter(
            radius=tube_radius, number_of_sides=20, vary_radius="vary_radius_off",
        )
        configure_input_data(pt, pd)
        m = tvtk.PolyDataMapper(input_connection=pt.output_port)
        a = tvtk.Actor(mapper=m)
        a.property.color = 0, 0, 0
        a.property.specular = 0.3
        return a

    v = info["verts4x4"]  # arranged in 2 rectangles of 4 verts

    actors = []

    # bottom rect
    actors.append(make_2_vert_tube(v[0], v[1]))
    actors.append(make_2_vert_tube(v[1], v[2]))
    actors.append(make_2_vert_tube(v[2], v[3]))
    actors.append(make_2_vert_tube(v[3], v[0]))

    # top rect
    actors.append(make_2_vert_tube(v[4], v[5]))
    actors.append(make_2_vert_tube(v[5], v[6]))
    actors.append(make_2_vert_tube(v[6], v[7]))
    actors.append(make_2_vert_tube(v[7], v[4]))

    # verticals
    actors.append(make_2_vert_tube(v[0], v[4]))
    actors.append(make_2_vert_tube(v[1], v[5]))
    actors.append(make_2_vert_tube(v[2], v[6]))
    actors.append(make_2_vert_tube(v[3], v[7]))

    return actors


def get_mayavi_cubic_arena_source(engine, info=None):

    v = info["verts4x4"]  # arranged in 2 rectangles of 4 verts

    points = v
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

    if 0:
        import enthought.mayavi.tools.mlab as mlab

        for lineseg in lines:
            p0 = points[lineseg[0]]
            p1 = points[lineseg[1]]
            x = np.array([p0[0], p1[0]])
            y = np.array([p0[1], p1[1]])
            z = np.array([p0[2], p1[2]])
            mlab.plot3d(x, y, z)

    # polys = numpy.arange(0, len(points), 1, 'l')
    # polys = numpy.reshape(polys, (len(points), 1))
    pd = tvtk.PolyData(points=points, lines=lines)  # polys=polys,

    e = engine
    e.add_source(VTKDataSource(data=pd, name="cubic arena"))

    s = Surface()
    e.add_module(s)
