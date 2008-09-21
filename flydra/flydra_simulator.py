# Start the mainbrain and several fake cameras in a way that allows
# profiling and testing the whole shebang.

# Note, it would be also nice to do emulation bypassing the
# ProcessCamClass part of the stack -- no need to test background
# subtraction and so on...

import subprocess, threading, os, time, signal, sys, socket, errno
import Pyro.core
import flydra.common_variables
from optparse import OptionParser

class RemoteCameraSource(Pyro.core.ObjBase):
    # === Methods called locally ==================================
    def post_init(self,quit_now):
        self.quit_now = quit_now
        self.last_timestamps = {}
        self.last_framenumbers = {}
    # === thread boundary =========================================
    def listen(self,daemon):
        """thread mainloop"""
        while not self.quit_now.isSet():
            daemon.handleRequests(0.1) # block on select for n seconds
    def get_frame_size(self,id):
        return 640,480
    def get_last_timestamp(self,id):
        return self.last_timestamps[id]
    def get_last_framenumber(self,id):
        print 'self.last_framenumbers[id]',id,self.last_framenumbers[id]
        return self.last_framenumbers[id]
    def get_point_list(self,id):
        # implement synchronization stuff
        time.sleep(0.016)
        self.last_timestamps[id] = time.time()
        self.last_framenumbers[id] = self.last_framenumbers.get(id,-1)+1
        return [ (320.1, 240.1) ]

class ProcessRunner(object):
    def __init__( self, args, env=None ):
        self.popen = subprocess.Popen(args,env=env)
        self.args = args
        self.returncode = None
    def query(self, quit_now=False):
        if quit_now:
            # attempt to kill nicely...
            os.kill( self.popen.pid, signal.SIGTERM )
            while 1:
                if self.popen.poll():
                    # process ended
                    break
                time.sleep(0.1)
            if not self.popen.poll():
                # attempt to kill less nicely...
                os.kill( self.popen.pid, signal.SIGKILL )
            # kill child
            self.popen.wait()
            self.returncode = self.popen.returncode
        if self.popen.poll() is not None:
            # process ended
            self.returncode = self.popen.returncode
        return self.returncode

def start_mainbrain():
    newenv = {'REQUIRE_FLYDRA_TRIGGER':'0'}
    newenv.update( os.environ )
    ro = ProcessRunner( ['flydra_mainbrain'], env=newenv)
    return ro

def start_cameras(full_image_emulation=False):
    newenv = {}
    newenv.update( os.environ )
    N_cameras = 5
    cam_strs = []
    for i in range(N_cameras):
        port = 8430+i
        cam_strs.append( '<net %d 640 480>'%port )
        image_sources = os.pathsep.join(cam_strs)
    if full_image_emulation:
        args = ['flydra_camera_node',
                '--emulation-image-sources=%s'%image_sources,
                '--debug-acquire', # temporary
                '--wx',
                '--disable-ifi-warning',
                ]
    else:
        args = ['flydra_camera_node',
                '--simulate-point-extraction=%s'%image_sources,
                '--debug-acquire', # temporary
                '--wx',
                '--disable-ifi-warning', # temporary?
                ]

    ro = ProcessRunner( args, env=newenv)
    return ro

def wait_for_mainbrain_to_come_alive():
    # Give the main brain time to start.
    port = flydra.common_variables.mainbrain_port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while 1:
        try:
            s.connect(('localhost', port))
        except socket.error, err:
            if err.args[0] == errno.ECONNREFUSED: # connection refused
                continue
        else:
            break # it accepted so, mainbrain is alive
    s.close()

def main():

    parser = OptionParser(usage='%prog [options]',
                          version="%prog 0.1")
    parser.add_option("--full-images", action='store_true')
    (options, args) = parser.parse_args()

    full_image_emulation = options.full_images

    mb_runner = start_mainbrain()

    wait_for_mainbrain_to_come_alive()

    Pyro.core.initServer(banner=0)
    hostname = 'localhost'
    port = flydra.common_variables.emulated_camera_control
    # start Pyro server
    daemon = Pyro.core.Daemon(host=hostname,port=port)
    quit_event = threading.Event()
    rcs = RemoteCameraSource(); rcs.post_init(quit_event)
    URI=daemon.connect(rcs,'remote_camera_source')

    # create and start listen thread
    listen_thread=threading.Thread(target=rcs.listen,
                                   name='RemoteCameraSource-Thread',
                                   args=(daemon,))
    listen_thread.start()

    cam_runner = start_cameras(full_image_emulation=full_image_emulation)

    while 1:
        if cam_runner.query() is not None:
            returncode = cam_runner.returncode
            mb_runner.query(quit_now=True)
            break
        if mb_runner.query() is not None:
            returncode = mb_runner.returncode
            cam_runner.query(quit_now=True)
            break
        time.sleep(0.1)
    sys.exit(returncode)


