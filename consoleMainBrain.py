#!/usr/bin/env python
import os, threading, time, glob, traceback
import MainBrain
try:
    from msvcrt import getch, kbhit
except ImportError:
    from unix_keyboard import getch, kbhit

class App:
    def __init__(self):
        self.cam_ids = []
        self.main_brain = MainBrain.MainBrain()
        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening() # spawns MainBrain and listens in another thread
        
        self.timer_thread = threading.Thread(target=self.timer_func)
        self.timer_thread.quitnow = threading.Event()
        self.timer_thread.setDaemon(True)
        self.timer_thread.start()
        
    def timer_func(self):
        while not self.timer_thread.quitnow.isSet():
            self.main_brain.service_pending() # may call OnNewCamera, OnOldCamera, etc
            time.sleep(0.2)
        
    def output(self,data):
        print data
        
    def OnNewCamera(self,cam_id, scalar_control_info, fqdnport):
        self.output('new camera attached: '+cam_id)
        self.cam_ids.append( cam_id )
        self.cam_ids.sort()
    
    def OnOldCamera(self,cam_id):
        self.output('camera detached: '+cam_id)
        self.cam_ids.remove( cam_id )

    def confirm_quit(self):
        print 'Quit. Are you sure? (y/[n])'
        while 1:
            keystroke = getch().lower()
            if keystroke == 'y':
                print 'Yes, quitting'
                return True
            elif keystroke in ['n','\r','\n']:
                print 'No, not quitting'
                return False

    def _get_choice(self,choicename,options,current_selection=None):
        print 'Choose %s:'%(choicename)
        choice_dict = {}
        for key, value in options:
            print '  %s: %s'%(key,value),
            if current_selection==value:
                print '<-- current selection'
            else:
                print
            choice_dict[key] = value
        print 'Enter choice:',
        keystroke = getch().lower()
        #print keystroke,
        if keystroke in choice_dict:
            current_selection = choice_dict[keystroke]
        else:
            print 'Did not understand'
            return current_selection
        print '(selected %s: %s)'%(choicename,current_selection)
        return current_selection
    
    def get_active_camera(self,choice):
        options = [(str(i),cam_id) for i,cam_id in enumerate(self.cam_ids)]
        options.append( ('a','all') )
        return self._get_choice('camera',options,current_selection=choice)

    def display_camera_status(self,cam_id=None):
        if cam_id is None:
            cam_id = self.active_camera
            if cam_id == None:
                self.output('no camera connected/active')
                return
        if cam_id == 'all':
            for cam_id in self.cam_ids:
                self.display_camera_status(cam_id)
            return
        
        (image, fps, distorted_points,
         image_coords) = self.main_brain.get_last_image_fps(
            cam_id,distort_points_to_align_with_image=False)
        
        self.output('camera status: '+cam_id)
        sci=self.main_brain.get_scalarcontrolinfo(cam_id)
        if fps is None: fps_str = '?'
        else: fps_str = '%.1f'%fps
        self.output('  %s ~fps (%.3f max)'%(fps_str,sci['max_framerate']))
        w,h=sci['width'],sci['height']
        l,b,r,t=sci['roi']
        self.output('  %dx%d (LB: %d,%d RT: %d,%d)'%(w,h,l,b,r,t))

        for valname in ['brightness','gain','shutter']:
            val,minv,maxv = sci[valname]
            self.output('  %s: %d (in range %d-%d)'%(valname,val,minv,maxv))

        if sci['trigger_source']:
            self.output('  external trigger')
        else:
            self.output('  internal trigger')

        if sci['roi2']:
            self.output('  ROI2 (region for spatial moment calculation): near brightest pixel')
        else:
            self.output('  ROI2 (region for spatial moment calculation): whole image')

        self.output('  %.1f diff_threshold, %.1f clear_threshold'%(
            sci['diff_threshold'],sci['clear_threshold']))
        print sci

    def load_calibration(self):
        caldir = os.environ.get('HOME','')
        calib_dirs = glob.glob(os.path.join(caldir,u'Cal*'))
        options = [(str(i),cam_id) for i,cam_id in enumerate(calib_dirs)]
        calib_dir = self._get_choice('calibration directory',options)
        self.main_brain.load_calibration(calib_dir)
        self.output('loaded calibration '+calib_dir)
        
    def open_camera_settings(self):
        cfgdir = os.environ.get('HOME','')
        testdir = os.path.join(cfgdir,u'CameraSettings')
        if os.path.exists(testdir):
            cfgdir = testdir
        cfg_fnames = glob.glob(os.path.join(cfgdir,u'*.cfg'))
        
        options = [(str(i),cam_id) for i,cam_id in enumerate(cfg_fnames)]

        open_filename = self._get_choice('camera setting file',options)
        if not os.path.exists(open_filename):
            print repr(open_filename),'does not exist'
            return
        
        fd = open(open_filename,'rb')
        buf = fd.read()
        
        all_params = eval(buf)
        try:
            for cam_id, params in all_params.iteritems():
                for property_name, value in params.iteritems():
                    self.main_brain.send_set_camera_property(cam_id,property_name,value)
        except KeyError,x:
            self.output('%s: %s'%(str(x.__class__),str(x)))
            return
        self.output('loaded camera setting file '+open_filename)
        
    def mainloop(self):
        quitnow = False
        self.active_camera = None
        self.output('\n')
        self.output("Press '?' for help\n")
        while not quitnow:
            # process keyhits
            if kbhit():
                try:
                    keystroke = getch().lower()
                    if keystroke == 'c':
                        self.active_camera = self.get_active_camera(self.active_camera)
                    elif keystroke == 'l':
                        self.load_calibration()
                    elif keystroke == 's':
                        self.display_camera_status()
                    elif keystroke == 'o':
                        self.open_camera_settings()
                    elif keystroke == 'q':
                        quitnow = self.confirm_quit()
                    elif keystroke == '?':
                        self.output('c - switch active Camera')
                        self.output('s - display camera Status')
                        self.output('l - Load calibration')
                        self.output('o - Open camera settings')
                        self.output('q - Quit')                
                        self.output('? - help')
                    elif keystroke == '\n':
                        self.output('')
                    else:
                        self.output('did not understand keystroke:'+repr(keystroke))
                except:
                    traceback.print_exc()
            else:
                time.sleep(0.05) # wait 50 msec
                
        self.timer_thread.quitnow.set()
        self.main_brain.quit()
        self.timer_thread.join(0.1) # timeout of 0.1 sec

def main():
    app = App()
    app.mainloop()
    
if __name__ == '__main__':
    main()
