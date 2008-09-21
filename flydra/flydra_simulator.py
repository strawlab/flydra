# Start the mainbrain and 5 fake cameras in a way that allows
# profiling and testing the whole shebang.

import subprocess, threading, os, time, signal

def run_process( args, env=None, quit_event=None, sleep_interval=0.1 ):
    popen = subprocess.Popen(args,env=env)
    while 1:
        print 'process',args
        time.sleep(sleep_interval)
        if quit_event.isSet():
            print 'killing process running',args
            # attempt to kill nicely...
            os.kill( popen.pid, signal.SIGTERM )
            while 1:
                if popen.poll():
                    # process ended
                    break
                time.sleep(0.1)
            if not popen.poll():
                # attempt to kill less nicely...
                os.kill( popen.pid, signal.SIGKILL )
            # kill child
            popen.wait()
            break
        if popen.poll():
            print 'process ended',args
            # process ended
            break
    return popen.returncode

def run_mainbrain(quit_event):
    print 'calling mainbrain'
    newenv = {'REQUIRE_FLYDRA_TRIGGER':'0'}
    newenv.update( os.environ )
    run_process( ['flydra_mainbrain'], env=newenv, quit_event=quit_event )
    #subprocess.check_call(
    #                      env=newenv)
    print 'done with mainbrain'

def run_cameras(quit_event):
    print 'calling cameras'
    newenv = {}
    newenv.update( os.environ )
    N_cameras = 5
    cam_strs = []
    for i in range(N_cameras):
        port = 8430+i
        cam_strs.append( '<net %d>'%port )
    emulation_image_sources = ':'.join(cam_strs)
    run_process( ['flydra_camera_node',
                  '--emulation-image-sources=%s'%emulation_image_sources],
                 env=newenv, quit_event=quit_event )
    ## subprocess.check_call(['flydra_camera_node'],
    ##                       env=newenv)
    print 'done calling cameras'

def main():
    mbdone = threading.Event()
    mbt = threading.Thread(target=run_mainbrain,args=(mbdone,))
    mbt.start()

    # give the main brain a second to start?

    cdone = threading.Event()
    ct = threading.Thread(target=run_cameras,args=(cdone,))
    ct.start()

    while 1:
        if not ct.isAlive():
            mbdone.set()
            print 'ct done'
            mbt.join()
            break
        if not mbt.isAlive():
            cdone.set()
            print 'mb done'
            ct.join()
            break
        time.sleep(0.1)



