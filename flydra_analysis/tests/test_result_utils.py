import pkg_resources
import tables

import flydra_analysis.analysis.result_utils as result_utils

by_version = {'0.4.28': pkg_resources.resource_filename('flydra_analysis.a2','sample_datafile-v0.4.28.h5'),
              }

def test_read_header():
    for expected_version in by_version:
        fname = by_version[expected_version]
        with tables.open_file(fname,mode='r') as h5:
            parsed = result_utils.read_textlog_header(h5)
            actual_version = parsed['flydra_version']
            assert actual_version==expected_version
