import aggdraw
import numpy as np
import scipy.misc
import warnings


def _flat_to_xy(coords_in):
    coords_in = np.asarray(coords_in)
    xin = coords_in[0::2]
    yin = coords_in[1::2]
    return xin, yin


class Xform(object):
    def transform_image(self, im):
        arr = scipy.misc.fromimage(im)
        arr_out = self._transform_image_dowork(arr)
        im_out = scipy.misc.toimage(arr_out)
        return im_out

    def __call__(self, coords_in):
        xin, yin = _flat_to_xy(coords_in)
        xout, yout = self._dowork(xin, yin)
        tmp = np.array([xout, yout]).T
        coords_out = tmp.ravel()
        return coords_out


class XformIdentity(Xform):
    def _transform_image_dowork(self, arr):
        return arr

    def _dowork(self, xin, yin):
        return xin, yin


class XformFlipY(Xform):
    def __init__(self, ymax=None):
        assert ymax is not None
        self.ymax = ymax
        super(XformFlipY, self).__init__()

    def _transform_image_dowork(self, arr):
        h = arr.shape[0]
        assert self.ymax == (h - 1)
        arr_out = np.flipud(arr)
        return arr_out

    def _dowork(self, xin, yin):
        return xin, self.ymax - yin


class XformRotate180(Xform):
    def __init__(self, xmax=None, ymax=None):
        assert xmax is not None
        assert ymax is not None
        self.xmax = xmax
        self.ymax = ymax
        super(XformRotate180, self).__init__(self)

    def _transform_image_dowork(self, arr):
        h, w = arr.shape[:2]
        assert self.xmax == (w - 1)
        assert self.ymax == (h - 1)
        arr_tmp = np.rot90(arr)
        arr_out = np.rot90(arr_tmp)
        return arr_out

    def _dowork(self, xin, yin):
        return self.xmax - xin, self.ymax - yin


class CoordShiftDraw(object):
    def __init__(self, im, xform):
        self._xform = xform
        self._im = self._xform.transform_image(im)
        self._draw = aggdraw.Draw(self._im)

    def get_image(self):
        return self._im

    def text(self, coords_in, *args, **kwargs):
        warnings.warn(
            "using CoordShiftDraw.text() instead of text_smartshift()", stacklevel=2
        )
        coords_out = self._xform(coords_in)  # make list, take first element
        return self._draw.text(coords_out, *args, **kwargs)

    def text_noxform(self, coords_in, *args, **kwargs):
        return self._draw.text(coords_in, *args, **kwargs)

    def text_smartshift(self, coords_in, ref_coords, *args, **kwargs):
        rx, ry = _flat_to_xy(ref_coords)
        x, y = _flat_to_xy(coords_in)
        dx = x - rx
        dy = y - ry
        center_coords = self._xform(ref_coords)
        cx, cy = _flat_to_xy(center_coords)
        assert len(cx.shape) == 1
        allx = cx + dx
        ally = cy + dy
        coords_out = (np.array([allx, ally]).T).ravel()
        # coords_out = self._xform(coords_in) # make list, take first element
        return self._draw.text(coords_out, *args, **kwargs)

    def textsize(self, *args, **kwargs):
        return self._draw.textsize(*args, **kwargs)

    def flush(self):
        return self._draw.flush()

    def ellipse(self, coords_in, *args, **kwargs):
        coords_out = self._xform(coords_in)  # make list, take first element
        return self._draw.ellipse(coords_out, *args, **kwargs)

    def line(self, coords_in, *args, **kwargs):
        coords_out = self._xform(coords_in)  # make list, take first element
        return self._draw.line(coords_out, *args, **kwargs)
