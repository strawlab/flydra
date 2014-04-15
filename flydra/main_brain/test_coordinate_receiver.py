import time
import Queue

from coordinate_receiver import CoordinateProcessor

import roslib
roslib.load_manifest('rospy')
import rospy

class FakeTriggerDevice():
    def wait_for_estimate(self):
        return

class FakeMainBrain:
    def __init__(self):
        self.hostname = 'localhost'
        self.queue_error_ros_msgs = Queue.Queue()
        self.trigger_device = FakeTriggerDevice()

def test_coordinate_receiver1():
    rospy.init_node('test_coordinate_receiver1',disable_signals=True)
    mb = FakeMainBrain()
    coord_processor = CoordinateProcessor(mb,
                                          save_profiling_data=False,
                                          debug_level=0,
                                          show_overall_latency=False,
                                          show_sync_errors=False,
                                          max_reconstruction_latency_sec=0.3,
                                          max_N_hypothesis_test=3)
    coord_processor.daemon = True
    coord_processor.start()
    time.sleep(5.0)
    coord_processor.quit()
