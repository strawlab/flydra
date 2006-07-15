import os, sys, subprocess

def main():
    flydra_dir=os.path.split(os.path.abspath(__file__))[0]
    flydra_file = os.path.join(flydra_dir,'flydra_camera_node.py')

    env={'CAM_IFACE_DUMMY':'1',
         'FLYDRA_BENCHMARK':'1',
         }

    if 1:
        os.environ.update(env)
        execfile(flydra_file,globals(),globals())
    else:
        args = [sys.executable,flydra_file]
        sub = subprocess.Popen(args,env=env)

if __name__=='__main__':
    main()
