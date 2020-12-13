from flydra_analysis.a2.calculate_reprojection_errors import \
     calculate_reprojection_errors, print_summarize_file
import flydra_analysis.a2.core_analysis as core_analysis

import os, tempfile, shutil
import pkg_resources

DATAFILE2D = pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile-v0.4.28.h5')
DATAFILE3D = pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile-v0.4.28.h5.retracked.h5')

def test_calculate_reprojection_errors():
    for from_source in ['smoothed', 'ML_estimates']:
        yield check_calculate_reprojection_errors, from_source

def check_calculate_reprojection_errors(from_source):
    tmpdir = tempfile.mkdtemp()
    try:
        outfile = os.path.join(tmpdir,'retracked.h5')
        calculate_reprojection_errors(h5_filename=DATAFILE2D,
                                      output_h5_filename=outfile,
                                      kalman_filename=DATAFILE3D,
                                      from_source=from_source,
                                      )
        assert os.path.exists(outfile)
        print_summarize_file(outfile)
        # XXX FIXME add some test beyond just running and printing it.

        # close open files
        ca = core_analysis.get_global_CachingAnalyzer()
        ca.close()
    finally:
        shutil.rmtree(tmpdir)
