from flydra.trigger import Device

# may need to do this: mount -t none /proc/bus/usb /dap386chrt/proc/bus/usb -o bind

import cam_iface_choose
global cam_iface
cam_iface = None

import numpy as nx
import time, sys, os
from optparse import OptionParser

import threading

def main():
    global cam_iface
    usage_lines = ['%prog [options]',
                   '',
                   '  available wrappers and backends:']
    
    for wrapper,backends in cam_iface_choose.wrappers_and_backends.iteritems():
        for backend in backends:
            usage_lines.append('    --wrapper %s --backend %s'%(wrapper,backend))
    del wrapper, backend # delete temporary variables
    usage = '\n'.join(usage_lines)
    
    parser = OptionParser(usage)
    
    parser.add_option("--wrapper", dest="wrapper", type='string',
                      help="cam_iface WRAPPER to use",
                      metavar="WRAPPER")
    
    parser.add_option("--backend", dest="backend", type='string',
                      help="cam_iface BACKEND to use",
                      metavar="BACKEND")
    
    parser.add_option("--mode-num", type="int",
                      help="mode number")
    
    (options, args) = parser.parse_args()
    
    if not options.wrapper:
        print 'WRAPPER must be set'
        parser.print_help()
        return
    
    if not options.backend:
        print 'BACKEND must be set'
        parser.print_help()
        return
    
    cam_iface = cam_iface_choose.import_backend( options.backend, options.wrapper )
    
    print 'options.mode_num',options.mode_num

    if options.mode_num is not None:
        mode_num = options.mode_num
    else:
        mode_num = 0
    doit(mode_num=mode_num)

def save_func( fly_movie, save_queue ):
    while 1:
        fnt = save_queue.get()
        frame,timestamp = fnt
        fly_movie.add_frame(frame,timestamp)
        if 1:
            import numpy
            f16 = numpy.fromstring(frame.tostring(),dtype=numpy.uint16)
            f16.shape = frame.shape[0], frame.shape[1]/2
            print
            print 'f16[:5,:5]'
            print f16[:5,:5]
            print

class Grabber:
    def __init__(self):
        # called from main thread
        self.last_pytimestamp = None
        self.last_pytimestamp_lock = threading.Lock()
        
        self.framecount = 0
        self.framecount_lock = threading.Lock()
    
    def run(self,device_num=0,mode_num=0,num_buffers=30):
        # runs in own thread
        global cam_iface

        num_modes = cam_iface.get_num_modes(device_num)
        for this_mode_num in range(num_modes):
            mode_str = cam_iface.get_mode_string(device_num,this_mode_num)
            print 'mode %d: %s'%(this_mode_num,mode_str)

        print 'choosing mode %d'%(mode_num,)

        cam = cam_iface.Camera(device_num,num_buffers,mode_num)
        cam.set_trigger_mode_number(1)
        print 'using ',cam.get_trigger_mode_string(cam.get_trigger_mode_number())
        cam.start_camera()
        n_frames = 0
        last_fps_print = time.time()
        last_fno = None
        while 1:
            try:
                buf = nx.asarray(cam.grab_next_frame_blocking())
            except cam_iface.FrameDataMissing:
                sys.stdout.write('M')
                sys.stdout.flush()            
                continue
            now = time.time()
            
            timestamp = cam.get_last_timestamp()

            fno = cam.get_last_framenumber()
            
            self.framecount_lock.acquire()
            self.framecount += 1
            self.framecount_lock.release()
            
            self.last_pytimestamp_lock.acquire()
            # switch between "now" and "timestamp" to get time.time or
            # backend's timestamp
            self.last_pytimestamp = now 
            self.last_pytimestamp_lock.release()

            if last_fno is not None:
                skip = (fno-last_fno)-1
                if skip != 0:
                    print 'WARNING: skipped %d frames'%skip
        ##    if n_frames==50:
        ##        print 'sleeping'
        ##        time.sleep(10.0)
        ##        print 'wake'
            last_fno=fno
            now = time.time()
            sys.stdout.write('.')
            sys.stdout.flush()
            n_frames += 1

            t_diff = now-last_fps_print
            if t_diff > 1.0:
                fps = n_frames/t_diff
                print "%.1f fps"%fps
                last_fps_print = now
                n_frames = 0


    def get_last_timestamp(self):
        # called from main thread
        self.last_pytimestamp_lock.acquire()
        ts = self.last_pytimestamp
        self.last_pytimestamp_lock.release()
        return ts
    
    def get_framecount(self):
        self.framecount_lock.acquire()
        fc = self.framecount
        self.framecount_lock.release()
        return fc
    
def doit(device_num=0,mode_num=0,num_buffers=30):

    start_fps = 80.0
    dev = Device()

    grabber = Grabber()

    gt = threading.Thread( target=grabber.run, args=(device_num,mode_num,num_buffers) )
    gt.setDaemon(True)
    gt.start()

    while 1:
        # start camera, ensure we get some frames
        dev.set_carrier_frequency(start_fps) # hz
        print 'make sure you see %f fps'%start_fps
        time.sleep(6)

        # stop camera, wait until no more frames are coming
        print 'stopping camera'
        dev.set_carrier_frequency(0)
        time.sleep(1)
        fc = grabber.get_framecount()
        time.sleep(.3)
        fc2 = grabber.get_framecount()

        try:
            assert fc == fc2
        except:
            print 'fc',fc
            print 'fc2',fc2
            raise

        # send single trigger pulse
        print 'sending single pulse'
        send_time = time.time()
        dev.trigger_once()
        back_time = time.time()
        time.sleep(1)

##        fc3 = grabber.get_framecount()

##        assert fc3 == fc + 1

        acquire_time = grabber.get_last_timestamp()

        max_latency = acquire_time - send_time
        min_latency = acquire_time - back_time

        print
        print
        print "using cam_iface's timestamp: min latency %.2f msec, max latency %.2f msec"%(min_latency*1e3, max_latency*1e3),'<','-'*20
        print
        print

if __name__=='__main__':
    main()
