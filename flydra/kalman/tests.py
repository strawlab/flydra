import unittest
import flydra_tracker, dynamic_models
import flydra.reconstruct
import pkg_resources

class TestTracker(unittest.TestCase):
    def setUp(self):
        caldir = pkg_resources.resource_filename('flydra','sample_calibration')
        self.reconstructor = flydra.reconstruct.Reconstructor(caldir)

    def test_encode_decode_roundtrip(self):
        if 1:
            # not completed. TODO - implement
            return
        models = dynamic_models.create_dynamic_model_dict(dt= 0.01)
        for model_name in models.keys():
            kalman_model = models[ model_name ]

            x = flydra_tracker.Tracker(self.reconstructor,
                                       kalman_model=kalman_model,
                                       )
            n_packets = 3
            n_3d_objs = 4
            for i in range(n_packets):
                corrected_framenumber = i
                timestamp = time.time()
                ss = kalman_model['ss']
                for j in range(n_3d_objs):
                    state_vecs = numpy.random.randn( ss )
                    meanP = 1e-4

            data_packet1 = x.encode_data_packet(fno1,timestamp1,timestamp1)
            data_packet2 = x.encode_data_packet(fno2,timestamp2,timestamp2)

            in_packets = [data_packet1, data_packet2]
            super_packet = encode_super_packet( in_packets )
            for i,decoded_packet in enumerate(decode_super_packet(super_packet)):
                orig_packet = in_packets[i]

                (corrected_framenumber, acquire_timestamp,
                 reconstruction_timestamp, state_vecs, meanP) = \
                 decode_data_packet( decoded_packet )


def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestTracker),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
