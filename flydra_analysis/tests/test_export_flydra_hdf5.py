import flydra_analysis.a2.data2smoothed
import os, tempfile, shutil
import pkg_resources

DATAFILE2D = pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile-v0.4.28.h5')
#DATAFILE3D = pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile.h5')
DATAFILE3D = pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile-v0.4.28.h5.retracked.h5')

def test_export_flydra_hdf5():
    for from_source in ['smoothed', 'ML_estimates']:
        yield check_export_flydra_hdf5, from_source

def check_export_flydra_hdf5(from_source):
    tmpdir = tempfile.mkdtemp()
    try:
        outfile = os.path.join(tmpdir,'exported.h5')
        flydra_analysis.a2.data2smoothed.convert(DATAFILE3D,outfile,
                                        file_time_data=DATAFILE2D,
                                        hdf5=True)
        # XXX FIXME add some test beyond just running it.
    finally:
        shutil.rmtree(tmpdir)
