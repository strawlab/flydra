import os, sys#, subprocess
#import execmodule

def main():
    flydra_dir=os.path.split(os.path.abspath(__file__))[0]
    flydra_file = os.path.join(flydra_dir,'flydra_camera_node.py')

    env={'CAM_IFACE_DUMMY':'1',
         'FLYDRA_BENCHMARK':'1',
         }

    if 1:
        os.environ.update(env)
        g = globals()
        g.update({'__name__':'__main__'})
        lc=g
        sys.argv[0] = flydra_file
        if len(sys.argv)>1:
            del sys.argv[1:]
        sys.argv.extend( '--wrapper dummy --backend dummy'.split() )
        execfile(flydra_file,g,lc)
    else:
        args = [sys.executable,flydra_file]
        sub = subprocess.Popen(args,env=env)

if __name__=='__main__':
    main()
