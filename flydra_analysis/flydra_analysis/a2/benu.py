"""object oriented, high-level wrapper of the (some of) cairo graphics library

What is a benu? A benu is a purple heron, a species of Egyptian bird
thought to be the phoenix.

"""
from __future__ import division, with_statement
from __future__ import absolute_import

import cairo
import os, warnings
import numpy as np
import contextlib
from .benu_colormaps import cmaps

D2R = np.pi / 180.0


def numpy2cairo(raw, cmap=None):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8:
        raise ValueError("only uint8 dtype is supported")
    if raw.ndim == 2:
        if cmap == None:  # gray
            brga = np.ndarray(shape=(raw.shape[0], raw.shape[1], 4), dtype=np.uint8)
            brga[:, :, 0] = raw
            brga[:, :, 1] = raw
            brga[:, :, 2] = raw
            brga[:, :, 3].fill(255)
        else:
            brga = cmaps[cmap][raw]
    elif raw.ndim == 3:
        if raw.shape[2] == 3:
            brga = np.ndarray(shape=(raw.shape[0], raw.shape[1], 4), dtype=np.uint8)
            brga[:, :, 0] = raw[:, :, 2]
            brga[:, :, 1] = raw[:, :, 1]
            brga[:, :, 2] = raw[:, :, 0]
            brga[:, :, 3].fill(255)
        elif raw.shape[2] == 4:
            brga = np.ndarray(shape=(raw.shape[0], raw.shape[1], 4), dtype=np.uint8)
            brga[:, :, 0] = raw[:, :, 2]
            brga[:, :, 1] = raw[:, :, 1]
            brga[:, :, 2] = raw[:, :, 0]
            brga[:, :, 3] = raw[:, :, 3]
        else:
            raise ValueError("if 3d array is given, it must be RGB or RGBA")
    else:
        raise ValueError("only only 2d and 3d arrays are supported")
    in_surface = cairo.ImageSurface.create_for_data(
        brga, cairo.FORMAT_ARGB32, raw.shape[1], raw.shape[0], raw.shape[1] * 4
    )  # , raw.shape[0]*3)
    return in_surface


class ExternalSurfaceCanvas(object):
    def __init__(self, context, color_rgba=None):
        self._ctx = context
        if color_rgba is not None:
            self._ctx.save()
            self._ctx.set_source_rgba(*color_rgba)
            self._ctx.paint()
            self._ctx.restore()

    def imshow(self, im, l, b, filter="nearest", cmap=None):
        """show image im at location (l,b)

        filter can be one of ['best', 'bilinear', 'fast', 'gaussian',
        'good', 'nearest'].

        """
        # Get cairo filter type (e.g. "cario.FILTER_NEAREST")
        cfilter = getattr(cairo, "FILTER_" + filter.upper())

        # Get cairo surface
        in_surface = numpy2cairo(im, cmap=cmap)

        ctx = self._ctx  # shorthand
        # ctx.rectangle(l,b,im.shape[1],im.shape[0])
        ctx.save()
        ctx.set_source_surface(in_surface, l, b)
        ctx.get_source().set_filter(cfilter)
        ctx.paint()
        ctx.restore()

    def plot(self, xarr, yarr, color_rgba=None, close_path=False, linewidth=None):
        """line plot of xarr vs. yarr"""
        if np.any(np.isnan(xarr)) or np.any(np.isnan(yarr)):
            raise ValueError("cannot plot data with nans")

        if len(xarr) == 1:
            warnings.warn("benu plot() currently only plots line segments")
        assert len(xarr) == len(yarr)
        if len(xarr) == 0:
            return

        if color_rgba is None:
            color_rgba = (1, 1, 1, 1)
        ctx = self._ctx  # shorthand

        if linewidth is not None:
            orig_linewidth = ctx.get_line_width()
            ctx.set_line_width(linewidth)

        ctx.set_source_rgba(*color_rgba)
        ctx.move_to(xarr[0], yarr[0])
        for i in range(1, len(xarr)):
            ctx.line_to(xarr[i], yarr[i])
        if close_path:
            ctx.close_path()
        ctx.stroke()

        if linewidth is not None:
            ctx.set_line_width(orig_linewidth)

    def poly(self, xarr, yarr, color_rgba=None, edgewidth=0.0):
        if np.any(np.isnan(xarr)) or np.any(np.isnan(yarr)):
            raise ValueError("cannot plot data with nans")

        if color_rgba is None:
            color_rgba = (1, 1, 1, 1)
        ctx = self._ctx  # shorthand
        if edgewidth is not None:
            orig_linewidth = ctx.get_line_width()
            ctx.set_line_width(edgewidth)
        ctx.set_source_rgba(*color_rgba)
        ctx.move_to(xarr[0], yarr[0])
        for i in range(1, len(xarr)):
            ctx.line_to(xarr[i], yarr[i])
        ctx.close_path()
        ctx.fill()
        ctx.stroke()
        if edgewidth is not None:
            ctx.set_line_width(orig_linewidth)

    def scatter(self, xarr, yarr, color_rgba=None, radius=1.0, markeredgewidth=None):
        """scatter plot of xarr vs. yarr"""
        if color_rgba is None:
            color_rgba = (1, 1, 1, 1)
        ctx = self._ctx  # shorthand

        if markeredgewidth is not None:
            orig_linewidth = ctx.get_line_width()
            ctx.set_line_width(markeredgewidth)

        ctx.set_source_rgba(*color_rgba)
        for x, y in zip(xarr, yarr):
            if np.isnan(x) or np.isnan(y):
                continue

            # cairo_new_sub_path()
            ctx.new_sub_path()
            ctx.arc(x, y, radius, 0, 2 * np.pi)
        ctx.stroke()

        if markeredgewidth is not None:
            ctx.set_line_width(orig_linewidth)

    def text(self, text, x, y, color_rgba=None, font_size=10, shadow_offset=None):
        """draw text"""
        if np.isnan(x) or np.isnan(y):
            raise ValueError("cannot plot data with nans")

        if shadow_offset is not None:
            self.text(
                text,
                x + shadow_offset,
                y + shadow_offset,
                color_rgba=(0, 0, 0, 1),
                font_size=font_size,
            )
        if color_rgba is None:
            color_rgba = (0, 0, 0, 1)

        ctx = self._ctx  # shorthand

        ctx.set_source_rgba(*color_rgba)
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(font_size)

        ctx.move_to(x, y)
        ctx.show_text(text)

    def _get_matrix_for_transform(self, device_rect, user_rect, transform=None):
        user_l, user_b, user_w, user_h = user_rect
        user_r = user_l + user_w
        user_t = user_b + user_h
        device_l, device_b, device_w, device_h = device_rect
        if transform is None:
            transform = "orig"

        if transform == "orig":
            xscale = device_w / user_w
            yscale = device_h / user_h
            matrix = cairo.Matrix(
                xx=xscale,
                yx=0,
                xy=0,
                yy=yscale,
                x0=(device_l - xscale * user_l),
                y0=(device_b - yscale * user_b),
            )
        elif transform.startswith("rot "):
            theta_deg = float(transform[4:])
            theta = theta_deg * D2R

            # user to device transform
            rotation = cairo.Matrix(
                xx=np.cos(theta),
                xy=-np.sin(theta),
                x0=0.0,
                yx=np.sin(theta),
                yy=np.cos(theta),
                y0=0.0,
            )

            # calculate scaling to fit for arbitrary rotation
            user_verts = [
                (user_l, user_b),
                (user_r, user_b),
                (user_r, user_t),
                (user_l, user_t),
            ]
            rotated_user_verts = np.array(
                [rotation.transform_point(*v) for v in user_verts]
            )
            rotated_user_l = np.min(rotated_user_verts[:, 0])
            rotated_user_r = np.max(rotated_user_verts[:, 0])
            rotated_user_b = np.min(rotated_user_verts[:, 1])
            rotated_user_t = np.max(rotated_user_verts[:, 1])
            rotated_user_w = rotated_user_r - rotated_user_l
            rotated_user_h = rotated_user_t - rotated_user_b

            rotated_scale_x = device_w / rotated_user_w
            rotated_scale_y = device_h / rotated_user_h

            scale = cairo.Matrix(
                xx=rotated_scale_x, xy=0.0, x0=0.0, yx=0.0, yy=rotated_scale_y, y0=0.0
            )

            x0 = device_l - rotated_user_l * rotated_scale_x
            y0 = device_b - rotated_user_b * rotated_scale_y

            translate = cairo.Matrix(xx=1.0, xy=0.0, x0=x0, yx=0.0, yy=1.0, y0=y0)

            matrix = rotation * scale * translate

            if 0:
                # pretend the device has y going up from origin at bottom
                device_coord_flip = cairo.Matrix(
                    xx=1.0, xy=0.0, x0=0.0, yx=0.0, yy=-1.0, y0=0.0
                )
                device_coord_translate = cairo.Matrix(
                    xx=1.0, xy=0.0, x0=0.0, yx=0.0, yy=1.0, y0=device_h
                )
                matrix = matrix * device_coord_flip * device_coord_translate

        else:
            raise ValueError("unknown transform '%s'" % transform)
        return matrix

    @contextlib.contextmanager
    def set_user_coords(self, device_rect, user_rect, clip=True, transform=None):
        """enter a benu context with a user-defined coordinate system

        **Arguments**

        device_rect : 4 element tuple
            Specifies the coordinates in device space to draw into (l,b,w,h)
        user_rect : 4 element tuple
            Specifies the coordinates in arbitrary user defined space
            mapping into device space (l,b,w,h)

        **Optional keyword arguments**

        clip : boolean
            Whether to limit drawing within the device_rect
        transform : string
            how user_rect is transformed into device_rect. One of 'orig',
            'rot -90', 'rot 180', 'rot 90'.
        """
        user_l, user_b, user_w, user_h = user_rect
        orig_matrix = self._ctx.get_matrix()
        matrix = self._get_matrix_for_transform(
            device_rect, user_rect, transform=transform
        )
        self._ctx.set_matrix(matrix)
        if clip:
            self._ctx.save()
            self._ctx.rectangle(user_l, user_b, user_w, user_h)
            self._ctx.clip()
            self._ctx.new_path()
        try:
            yield
        finally:
            if clip:
                self._ctx.restore()
            self._ctx.set_matrix(orig_matrix)

    def get_transformed_point(self, x, y, device_rect, user_rect, transform=None):
        matrix = self._get_matrix_for_transform(
            device_rect, user_rect, transform=transform
        )
        return matrix.transform_point(x, y)


def test_benu():
    import tempfile

    tmp_fname = tempfile.mktemp(".png")
    canv = Canvas(tmp_fname, 1024, 1024)
    device_rect = (256, 256, 512, 512)
    ux0 = 0
    uy0 = 0
    uw = 50
    uh = 50
    ux1 = ux0 + uw
    uy1 = uy0 + uh
    user_rect = (ux0, uy0, uw, uh)

    # transform = 'rot 10'
    transform = "rot -45"
    # transform = 'orig'
    pts = [
        (0, 0),
        (5, 5),
        (30, 30),
        (45, 45),
        (1, 3),
        (6, 2),
    ]
    with canv.set_user_coords(device_rect, user_rect, transform=transform):
        for pt in pts:
            canv.scatter([pt[0]], [pt[1]])
        # draw boundary in user coords
        canv.plot(
            [ux0, ux0, ux1, ux1, ux0],
            [uy0, uy1, uy1, uy0, uy0],
            color_rgba=(1, 0, 0, 1),
        )

    if 1:
        # draw boundary of above coord system
        canv.plot(
            [
                device_rect[0],
                device_rect[0],
                device_rect[0] + device_rect[2],
                device_rect[0] + device_rect[2],
                device_rect[0],
            ],
            [
                device_rect[1],
                device_rect[1] + device_rect[3],
                device_rect[1] + device_rect[3],
                device_rect[1],
                device_rect[1],
            ],
            color_rgba=(0, 0.5, 0, 1),
        )

    # draw star
    canv.poly(
        [0, 50, 100, 0, 100, 0],
        [100, 0, 100, 50, 50, 100],
        color_rgba=(0, 0, 1, 1),
        edgewidth=5,
    )

    for pt in pts:
        x, y = canv.get_transformed_point(
            pt[0], pt[1], device_rect, user_rect, transform=transform
        )
        canv.text("%s, %s" % (pt[0], pt[1]), x, y)
    canv.save()

    # this test is broken...
    ## with c.set_user_coords(1,2):
    ##     pass


class Canvas(ExternalSurfaceCanvas):
    """A drawing surface which handles coordinate transforms"""

    def __init__(self, fname, width, height, **kwargs):
        self._output_ext = os.path.splitext(fname)[1].lower()
        if self._output_ext == ".pdf":
            output_surface = cairo.PDFSurface(fname, width, height)
        elif self._output_ext == ".svg":
            output_surface = cairo.SVGSurface(fname, width, height)
        elif self._output_ext == ".png":
            output_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        else:
            raise ValueError("unknown output extension %s" % self._output_ext)
        self._surf = output_surface
        context = cairo.Context(self._surf)
        self._fname = fname
        super(Canvas, self).__init__(context, **kwargs)

    def save(self):
        """save output to file"""
        if self._output_ext == ".png":
            self._surf.write_to_png(self._fname)
        else:
            self._ctx.show_page()
            self._surf.finish()

    def as_numpy(self):
        """save output to file"""
        assert self._output_ext == ".png"
        buf = self._surf.get_data()

        a = np.frombuffer(buf, np.uint8)
        a.shape = (self._surf.get_width(), self._surf.get_height(), 4)
        return a


if __name__ == "__main__":
    test_benu()
