import flydra.reconstruct
import pkg_resources

def test_failing_quartic_roots():
    fname = pkg_resources.resource_filename('flydra.a2','sample_calibration2.xml')
    R = flydra.reconstruct.Reconstructor(fname)
    cam_id = 'Basler_21426001'
    X3d = (-0.054036740422969506, 0.045115631117431117, -1.4063962542297526e-09)
    R.find2d( cam_id, X3d, distorted=True)
