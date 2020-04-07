import os, glob, time, sys, warnings, glob

try:
    import SCons
except ImportError as err:
    # Maybe this is related to
    # http://scons.tigris.org/issues/show_bug.cgi?id=1488
    maybe_scons = glob.glob(os.path.join(sys.prefix, "lib/scons*"))
    maybe_scons.sort()
    if len(maybe_scons):
        toadd = maybe_scons[-1]
        sys.path.append(toadd)
        warnings.warn("hack to import SCons: adding %s to path" % toadd)
        import SCons
        import SCons.Node
        import SCons.Builder
    else:
        raise

import flydra_analysis.a2.auto_discover_ufmfs
from flydra_analysis.a2.orientation_ekf_fitter import is_orientation_fit

_my_normcase = SCons.Node.FS._my_normcase


class UFMF(SCons.Node.FS.File):
    pass


# Potentional calibration sources


class CalibrationNodeMixin:
    pass


class FlydraCalibrationXMLFile(SCons.Node.FS.File, CalibrationNodeMixin):
    pass


# Potentional calibration sources


class Flydra2DDistortedDataNodeMixin:
    pass


class Flydra3DKalmanizedDataNodeMixin:
    pass


class FlydraFitOriMixin:
    pass


class H5File(SCons.Node.FS.File):
    pass


class Flydra2DDistortedDataH5File(H5File, Flydra2DDistortedDataNodeMixin):
    pass


class Flydra3DKalmanizedDataH5File(H5File, Flydra3DKalmanizedDataNodeMixin):
    pass


class Flydra3DKalmanizedDataFitOriH5File(
    Flydra3DKalmanizedDataH5File, FlydraFitOriMixin
):
    pass


class Flydra2D3DH5File(
    H5File, Flydra3DKalmanizedDataNodeMixin, Flydra2DDistortedDataNodeMixin
):
    pass


# helper functions


def get_all_instances(source_list, klass):
    return [r for r in source_list if isinstance(r, klass)]


def get_single_instance(source_list, klass):
    filtered = get_all_instances(source_list, klass)
    if len(filtered) > 1:
        sys.stderr.write("instances were: %s\n" % str([str(f) for f in filtered]))
        raise ValueError("only one instance of %s may be set" % klass)
    if len(filtered) < 1:
        raise ValueError("one instance of %s must be specified" % klass)
    return filtered[0]


# convert string to Node instance


def flydra_source_factory(source_string):
    """convert filename to Node instance"""
    base, ext = os.path.splitext(source_string)
    abspath = os.path.abspath(source_string)
    dirname, fname = os.path.split(abspath)
    directory = SCons.Node.FS.default_fs.Dir(dirname)
    if ext.lower() == ".xml":
        result = FlydraCalibrationXMLFile(fname, directory, fs=SCons.Node.FS.default_fs)

    elif ext.lower() == ".ufmf":
        result = UFMF(fname, directory, fs=SCons.Node.FS.default_fs)

    elif ext.lower() == ".h5":
        import tables  # pytables

        h5 = tables.open_file(abspath, mode="r")
        if hasattr(h5.root, "data2d_distorted") and hasattr(
            h5.root, "kalman_estimates"
        ):
            if is_orientation_fit(abspath):
                raise NotImplementedError("cannot deal with this file type yet")
            result = Flydra2D3DH5File(fname, directory, fs=SCons.Node.FS.default_fs)
        elif hasattr(h5.root, "data2d_distorted"):
            result = Flydra2DDistortedDataH5File(
                fname, directory, fs=SCons.Node.FS.default_fs
            )
        elif hasattr(h5.root, "kalman_estimates"):
            if is_orientation_fit(abspath):
                result = Flydra3DKalmanizedDataFitOriH5File(
                    fname, directory, fs=SCons.Node.FS.default_fs
                )
            else:
                result = Flydra3DKalmanizedDataH5File(
                    fname, directory, fs=SCons.Node.FS.default_fs
                )
        else:
            result = H5File(fname, directory, fs=SCons.Node.FS.default_fs)
        h5.close()

    elif ext.lower() == ".kh5":  # old name for kalmanized H5 file
        result = Flydra3DKalmanizedDataH5File(
            fname, directory, fs=SCons.Node.FS.default_fs
        )
    else:
        # default behavior
        result = Entry(source_string)
    return result


def create_node(file_name, dir_node, result):
    """add node to directory (required for scons to know to build target"""
    result.diskcheck_match()
    dir_node.entries[_my_normcase(file_name)] = result
    dir_node.implicit = None


def data2d_distorted_target_factory(target_string):
    """convert filename to Node instance, add to directory"""
    dirname, fname = os.path.split(os.path.abspath(target_string))
    directory = SCons.Node.FS.default_fs.Dir(dirname)
    result = Flydra2DDistortedDataH5File(fname, directory, fs=SCons.Node.FS.default_fs)
    create_node(fname, directory, result)
    return result


def kalmanizedH5_target_factory(target_string):
    """convert filename to Node instance, add to directory"""
    dirname, fname = os.path.split(os.path.abspath(target_string))
    directory = SCons.Node.FS.default_fs.Dir(dirname)
    result = Flydra3DKalmanizedDataH5File(fname, directory, fs=SCons.Node.FS.default_fs)
    create_node(fname, directory, result)
    return result


def kalmanizedFixedOriH5_target_factory(target_string):
    """convert filename to Node instance, add to directory"""
    dirname, fname = os.path.split(os.path.abspath(target_string))
    directory = SCons.Node.FS.default_fs.Dir(dirname)
    result = Flydra3DKalmanizedDataFitOriH5File(
        fname, directory, fs=SCons.Node.FS.default_fs
    )
    create_node(fname, directory, result)
    return result


# create Builders


def generate_ImageBasedData2DH5(source, target, env, for_signature):
    ufmfs = get_all_instances(source, UFMF)
    orig_data2d = get_single_instance(source, Flydra2DDistortedDataNodeMixin)
    orig_data3d = get_single_instance(source, Flydra3DKalmanizedDataNodeMixin)
    assert len(source) == len(ufmfs) + 2

    assert len(target) == 1
    target_name = target[0]
    ufmf_names = os.pathsep.join([str(ufmf) for ufmf in ufmfs])  # get Scons path
    args = " ".join(env.get("ImageBasedData2DH5_args", []))
    return (
        "flydra_analysis_image_based_orientation "
        "--h5=%(orig_data2d)s --kalman=%(orig_data3d)s "
        "--ufmfs=%(ufmf_names)s --output-h5=%(target_name)s "
        "%(args)s" % locals()
    )


def generate_RetrackedData2DH5(source, target, env, for_signature):
    ufmfs = get_all_instances(source, UFMF)
    orig_data2d = get_single_instance(source, Flydra2DDistortedDataNodeMixin)
    assert len(source) == len(ufmfs) + 1

    assert len(target) == 1
    target_name = target[0]
    ufmf_names = os.pathsep.join([str(ufmf) for ufmf in ufmfs])  # get Scons path
    args = " ".join(env.get("RetrackedData2DH5_args", []))
    return (
        "flydra_analysis_retrack_movies "
        "--h5=%(orig_data2d)s "
        "--ufmfs=%(ufmf_names)s --output-h5=%(target_name)s "
        "%(args)s" % locals()
    )


def generate_KalmanizedH5(source, target, env, for_signature):
    cal_source = get_single_instance(source, CalibrationNodeMixin)
    h5_source = get_single_instance(source, Flydra2DDistortedDataNodeMixin)
    assert len(source) == 2  # only the calibration and 2d data file

    assert len(target) == 1
    kh5_target = target[0]
    args = " ".join(env.get("KalmanizedH5_args", []))
    return (
        "flydra_kalmanize %(h5_source)s --reconstructor=%(cal_source)s "
        "--dest-file=%(kh5_target)s %(args)s" % locals()
    )


def generate_kalmanizedFixedOriH5(source, target, env, for_signature):
    kh5_source = get_single_instance(source, Flydra3DKalmanizedDataNodeMixin)
    h5_source = get_single_instance(source, Flydra2DDistortedDataNodeMixin)
    assert len(source) == 2  # only the 3d and 2d data files

    assert len(target) == 1
    kh5_target = target[0]
    args = " ".join(env.get("KalmanizedOriFitH5_args", []))
    return (
        "flydra_analysis_orientation_ekf_fitter --h5 %(h5_source)s -k %(kh5_source)s "
        "--output-h5 %(kh5_target)s %(args)s" % locals()
    )


ImageBasedData2DH5_Builder = SCons.Builder.Builder(
    generator=generate_ImageBasedData2DH5,
    target_factory=data2d_distorted_target_factory,
    source_factory=flydra_source_factory,
)

RetrackedData2DH5_Builder = SCons.Builder.Builder(
    generator=generate_RetrackedData2DH5,
    target_factory=data2d_distorted_target_factory,
    source_factory=flydra_source_factory,
)

KalmanizedH5_Builder = SCons.Builder.Builder(
    generator=generate_KalmanizedH5,
    target_factory=kalmanizedH5_target_factory,
    source_factory=flydra_source_factory,
)

KalmanizedFixedOriH5_Builder = SCons.Builder.Builder(
    generator=generate_kalmanizedFixedOriH5,
    target_factory=kalmanizedFixedOriH5_target_factory,
    source_factory=flydra_source_factory,
)
