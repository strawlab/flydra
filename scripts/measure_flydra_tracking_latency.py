#!/usr/bin/env python
import time
import argparse
import Queue
import threading

import roslib
roslib.load_manifest('triggerbox')
roslib.load_manifest('ros_flydra')
roslib.load_manifest('std_srvs')

import rospy
from triggerbox.triggerbox_client import TriggerboxClient
from triggerbox.msg import AOutConfirm
import ros_flydra.msg
import std_msgs.msg
import std_srvs.srv

class FramesPerSecondWaitClass(threading.Thread):
    def __init__(self,wq,trigger_host_node):
        self.wq = wq
        topic_name = trigger_host_node + '/expected_framerate'
        self.sub = rospy.Subscriber(topic_name,
                                    std_msgs.msg.Float32,
                                    self.got_fps)
        super(FramesPerSecondWaitClass,self).__init__()
        self.daemon=True
        self.start()

    def run(self):
        while 1:
            time.sleep(0.2)
    def got_fps(self,msg):
        self.wq.put(msg.data)

class AOutWaitClass(threading.Thread):
    def __init__(self,wq,trigger_host_node):
        self.wq = wq
        topic_name = trigger_host_node+'/aout_confirm'
        self.sub = rospy.Subscriber(topic_name,
                                    AOutConfirm,
                                    self.got_confirmation)
        super(AOutWaitClass,self).__init__()
        self.daemon=True
        self.start()

    def run(self):
        while 1:
            time.sleep(0.2)
    def got_confirmation(self,msg):
        self.wq.put(msg)

class FlydraWaitClass(threading.Thread):
    def __init__(self,wq,mainbrain_node):
        self.wq = wq
        topic_name = mainbrain_node+'/super_packets'
        self.sub = rospy.Subscriber(topic_name,
                                    ros_flydra.msg.flydra_mainbrain_super_packet,
                                    self.got_superpacket)
        super(FlydraWaitClass,self).__init__()
        self.daemon=True
        self.start()

    def run(self):
        while 1:
            time.sleep(0.2)
    def got_superpacket(self,msg):
        assert len(msg.packets)==1
        packet = msg.packets[0]
        reconstruction_stamp = packet.reconstruction_stamp.to_sec()
        acquire_stamp = packet.acquire_stamp.to_sec()
        self.wq.put( (reconstruction_stamp, acquire_stamp ) )

def send_and_wait(wq,trig,v0,v1,timeout=0.8):
    trig.set_aout_ab_volts(v0,v1)
    raw = wq.get(True,timeout)
    framestamp = raw.pulsenumber + raw.fraction_n_of_255/255.0

    return trig.framestamp2timestamp(framestamp)

def main():
    parser = argparse.ArgumentParser()
    rospy.init_node('measure_flydra_tracking_latency')
    argv = rospy.myargv()
    args = parser.parse_args(argv[1:])

    DEFAULT_TRIGGER_HOST_NODE = '/triggerbox_host'
    trigger_host_node = rospy.resolve_name(DEFAULT_TRIGGER_HOST_NODE)
    if trigger_host_node == DEFAULT_TRIGGER_HOST_NODE:
        rospy.logwarn('using default trigger host name %r'%trigger_host_node)
    else:
        rospy.loginfo('using trigger host name %r'%trigger_host_node)

    DEFAULT_MAINBRAIN_NODE = '/flydra_mainbrain'
    mainbrain_node = rospy.resolve_name(DEFAULT_MAINBRAIN_NODE)
    if mainbrain_node == DEFAULT_MAINBRAIN_NODE:
        rospy.logwarn('using default mainbrain name %r'%mainbrain_node)
    else:
        rospy.loginfo('using mainbrain name %r'%mainbrain_node)

    aout_queue = Queue.Queue()

    # Start the subscribers before connecting to the triggerbox to
    # increase likelihood of connecting with publisher for AOUT
    # publication. (I.e. avoid a race whereby it takes ROS a while to
    # setup a subscriber.)

    aout_waiter = AOutWaitClass(aout_queue,trigger_host_node)

    flydra_queue = Queue.Queue()

    flydra_waiter = FlydraWaitClass(flydra_queue, mainbrain_node )

    no_bg = rospy.ServiceProxy(mainbrain_node+'/stop_collecting_background',
                               std_srvs.srv.Empty)
    no_bg()
    rospy.loginfo('disabled running BG update')
    out_fname = 'latency.csv'
    rospy.loginfo('saving latency results to %s'%out_fname)

    fps_queue = Queue.Queue()
    fps_waiter = FramesPerSecondWaitClass(fps_queue, trigger_host_node)
    rospy.loginfo('waiting for FPS value...')
    fps = fps_queue.get()
    while 1:
        try:
            fps = fps_queue.get_nowait()
        except Queue.Empty:
            break
    rospy.loginfo('initial camera FPS of %r'%fps)

    trig = TriggerboxClient(host_node=trigger_host_node)
    trig.wait_for_estimate()

    with open(out_fname,mode='w') as fd:
        fd.write('%r,%r\n'%('measured_latency_msec','theoretical_latency_msec'))
        for i in range(10000):
            send_and_wait(aout_queue,trig,0,0)
            time.sleep(2.0)


            # clear any incoming flydra packet data
            while 1:
                try:
                    flydra_queue.get_nowait()
                except Queue.Empty:
                    break

            # check that no new data comes
            time.sleep(0.1)
            n_frames = 0
            while 1:
                try:
                    flydra_queue.get_nowait()
                    n_frames += 1
                except Queue.Empty:
                    break
            assert n_frames == 0

            aout_stamp = send_and_wait(aout_queue,trig,4,4)

            reconstruct_stamp, acquire_stamp = flydra_queue.get()

            # update fps if it changed
            while 1:
                try:
                    fps = fps_queue.get_nowait()
                    rospy.loginfo('camera FPS of %r'%fps)
                except Queue.Empty:
                    break

            acquire2_stamp = acquire_stamp + 1.0/fps
            measured_latency_msec = (reconstruct_stamp-aout_stamp)*1000.0
            theoretical_max_latency_msec = (reconstruct_stamp-acquire_stamp)*1000.0
            theoretical_min_latency_msec = (reconstruct_stamp-acquire2_stamp)*1000.0

            fd.write('%r,%r,%r\n'%(measured_latency_msec,theoretical_min_latency_msec,theoretical_max_latency_msec))
            fd.flush()
            time.sleep(0.2)

if __name__=='__main__':
    main()
