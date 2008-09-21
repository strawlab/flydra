# Start the mainbrain and several fake cameras in a way that allows
# profiling and testing the whole shebang.

import subprocess, os, time, signal, sys

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
        if self.popen.poll():
            # process ended
            self.returncode = self.popen.returncode
        return self.returncode

def start_mainbrain():
    newenv = {'REQUIRE_FLYDRA_TRIGGER':'0'}
    newenv.update( os.environ )
    ro = ProcessRunner( ['flydra_mainbrain'], env=newenv)
    return ro

def start_cameras():
    newenv = {}
    newenv.update( os.environ )
    N_cameras = 5
    cam_strs = []
    for i in range(N_cameras):
        port = 8430+i
        cam_strs.append( '<net %d>'%port )
    emulation_image_sources = ':'.join(cam_strs)
    ro = ProcessRunner( ['flydra_camera_node',
                         '--emulation-image-sources=%s'%emulation_image_sources],
                        env=newenv)
    return ro

def main():
    mb_runner = start_mainbrain()
    # give the main brain a second to start?

    cam_runner = start_cameras()

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


