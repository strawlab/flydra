import numpy
import numpy as np


def some_rough_negative_log_likelihood(
    pt_area=None, cur_val=None, mean_val=None, sumsqf_val=None
):
    # pt_area and sumsqf_val can be None

    scalar = numpy.isscalar(cur_val)

    cur_val = numpy.atleast_1d(cur_val)
    mean_val = numpy.atleast_1d(mean_val)

    if (
        np.all(cur_val == 0)
        and np.all(np.isnan(mean_val))
        and np.all(np.isnan(sumsqf_val))
    ):
        # This happens when using retracked images.
        raise RuntimeError(
            "should not gate data on appearance. "
            "Hint: use --disable-image-stat-gating"
        )

    if sumsqf_val is not None:
        sumsqf_val = numpy.atleast_1d(sumsqf_val)

    curmean = cur_val / mean_val

    n_std_certain = 100.0  # over threshold

    if sumsqf_val is not None:
        std = numpy.sqrt(abs(mean_val ** 2 - sumsqf_val))
        std = numpy.ma.masked_where(std == 0.0, std)
        absdiff = abs(cur_val - mean_val)
        n_std = absdiff / std

        # handle masked cases (where std==0)
        n_std.fill_value = n_std_certain
        n_std = n_std.filled()

        result_cond = ((curmean > 1.1) & (n_std > 5.0)) | (
            (curmean < 0.972) & (n_std > 4.71)
        )
    else:
        import warnings

        warnings.warn("computing point probability without STD information")
        result_cond = (curmean > 1.1) | (curmean < 0.972)

    result = numpy.where(result_cond, 0, numpy.inf)
    if scalar:
        result = result[0]
    return result
