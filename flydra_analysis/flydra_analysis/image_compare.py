import scipy.misc
import numpy as np


def get_fraction_similar(im1_filename, im2_filename, pix_diff_threshold=0.1):
    """Get fraction of pixels that are similar

    If all pixels are identical, return 1.0. If not, convert to
    grayscale (by taking the mean of the color and alpha channels, if
    present), count the number of pixels more different than
    pix_diff_threshold, and return the fraction of similar pixels.

    """
    im1 = scipy.misc.imread(im1_filename)
    im2 = scipy.misc.imread(im2_filename)
    if im1.ndim != im2.ndim:
        raise ValueError("images have different ndim")
    if im1.shape != im2.shape:
        raise ValueError("images have different shape")
    if np.allclose(im1, im2):
        # identical -- no more testing needed
        return True

    # maybe-3D absolute difference image
    di = abs(im1.astype(np.float64) - im2.astype(np.float64))

    # ensure 2D
    if di.ndim == 3:
        # flatten
        di2d = np.mean(di, axis=2)
    else:
        di2d = di

    # Count number of pixels more different than pix_diff_threshold.
    n_diff = np.sum(di2d > pix_diff_threshold)

    # Find fraction of all pixels
    n_total = di2d.shape[0] * di2d.shape[1]
    fraction_different = n_diff / float(n_total)
    fraction_same = 1.0 - fraction_different
    return fraction_same


def are_images_close(
    im1_filename, im2_filename, pix_diff_threshold=0.1, ok_fraction_threshold=0.99,
):
    """Compare whether two image files are very similar

    Calls get_fraction_similar() and returns True if the result is
    greater than or equal to ok_fraction_threshold, False otherwise.

    """
    fraction_similar = get_fraction_similar(
        im1_filename, im2_filename, pix_diff_threshold=pix_diff_threshold
    )
    # Compare with ok_fraction_threshold
    return fraction_similar >= ok_fraction_threshold
