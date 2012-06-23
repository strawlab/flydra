#!/usr/bin/env python
# $Id: $

import sys,threading, time, os, copy
import traceback
import optparse
import MainBrain

import roslib; roslib.load_manifest('rospy')
import rospy

class MainBrainApp():
    def __init__(self, *args, **kwargs):
        self._currently_recording_cams = []
        self.cameras = {}
        self.last_sound_time = 0

    def PreviewPerCamInit(self,cam_id, name):
        self.main_brain.send_set_camera_property(cam_id,name,value)
 
    def OnRecordRawStart(self):
        if len(self._currently_recording_cams) != 0:
            raise RuntimeError("currently recording!")
        
        raw_filename = "test.fmf"
        if raw_filename.endswith('.fmf'):
            bg_filename = raw_filename[:-4] + '_bg.fmf'
        else:
            bg_filename = raw_filename + '.bg.fmf'
        cam_ids = []
        for i in range(cam_choice.GetCount()):
            if cam_choice.IsChecked(i):
                cam_ids.append(cam_choice.GetString(i))
        if len(cam_ids)==0:
            return
        try:
            for cam_id in cam_ids:
                self.main_brain.start_recording(cam_id, raw_filename, bg_filename)
                self._currently_recording_cams.append(cam_id)
            print 'Recording started on %d cameras'% len(self._currently_recording_cams)
        except Exception, exc:
            try:
                for tmp_cam_id in self._currently_recording_cams[:]:
                    self.main_brain.stop_recording(tmp_cam_id)
                    self._currently_recording_cams.remove(tmp_cam_id)
            finally:
                print 'Failed to start recording (%s): see console' % cam_id
                raise exc

    def OnRecordRawStop(self,warn=True):
        if warn and not len(self._currently_recording_cams):
            print 'Not recording - cannot stop'
            return
        try:
            n_stopped = 0
            for cam_id in self._currently_recording_cams[:]:
                try:
                    self.main_brain.stop_recording(cam_id)
                except KeyError, x:
                    print '%s: %s'%(x.__class__,str(x)),
                self._currently_recording_cams.remove(cam_id)
                n_stopped+=1
            print 'Recording stopped on %d cameras' % n_stopped
        except:
            print 'Failed to stop recording: see console'
            raise


    def OnLoadCal(self):
        self.main_brain.load_calibration(calib_dir)

    def OnSnapshot(self,event,cam_id):
        if cam_id == '':
            return
        image, show_fps, points, image_coords = self.main_brain.get_last_image_fps(cam_id) # returns None if no new image
        if image is None:
            return

    def OnFakeSync(self):
        print 'sending fake sync command...'
        self.main_brain.fake_synchronize()

    def OnToggleDebugCameras(self):
        self.main_brain.set_all_cameras_debug_mode( event.IsChecked() )

    def OnStartSavingData(self):
        display_save_filename = time.strftime( 'DATA%Y%m%d_%H%M%S.h5' )
        save_dir = os.environ.get('HOME','')
        test_dir = os.path.join( save_dir, 'ORIGINAL_DATA' )
        if os.path.exists(test_dir):
            save_dir = test_dir
        save_filename = os.path.join( save_dir, display_save_filename )
        try:
            self.main_brain.start_saving_data(save_filename)
            print "Saving data to '%s'" % save_filename
        except:
            print "Error saving data to '%s', see console"%save_filename
            raise

    def OnStopSavingData(self):
        self.main_brain.stop_saving_data()
        print "Saving stopped"
        
    def OnSetROI(self, cam_id):
        lbrt = self.main_brain.get_roi(cam_id)
        width, height = self.main_brain.get_widthheight(cam_id)
        
        self.main_brain.send_set_camera_property(cam_id,'roi',lbrt)

    def attach_and_start_main_brain(self,main_brain):
        self.main_brain = main_brain
        self.main_brain.set_new_camera_callback(self.OnNewCamera)
        self.main_brain.set_old_camera_callback(self.OnOldCamera)
        self.main_brain.start_listening()

    def OnOpenCamConfig(self, open_filename='flydra_cameras.cfg'):
        fd = open(open_filename,'rb')
        buf = fd.read()
        all_params = eval(buf)
        try:
            for cam_id, params in all_params.iteritems():
                for property_name, value in params.iteritems():
                    self.main_brain.send_set_camera_property(cam_id,property_name,value)
        except KeyError, exc:
            print 'Error opening configuration data:%s' % exc
                    
    def OnSaveCamConfig(self, save_filename='flydra_cameras.cfg'):
        all_params = self.main_brain.get_all_params()
        fd = open(save_filename,'wb')
        fd.write(repr(all_params))
        fd.close()
        
    def OnStartCalibration(self, calib_dir):
        self.main_brain.start_calibrating(calib_dir)

    def OnStopCalibrating(self):
        self.main_brain.stop_calibrating()

    def OnQuit(self):
        self.main_brain.quit()
            
    def OnUpdateRawImages(self):
        for cam_id in self.cameras.keys():
            try:
                self.main_brain.request_image_async(cam_id)
            except KeyError: # no big deal, camera probably just disconnected
                pass

    def OnTimer(self):
        if not hasattr(self,'main_brain'):
            return # quitting
        self.main_brain.service_pending() # may call OnNewCamera, OnOldCamera, etc

        realtime_data=MainBrain.get_best_realtime_data()
        if realtime_data is not None:
            data3d,line3d,cam_ids_used,min_mean_dist=realtime_data
            if self.current_page == 'tracking':
                print 'x_pos % 8.1f'%data3d[0]
                print 'y_pos % 8.1f'%data3d[1]
                print 'z_pos % 8.1f'%data3d[2]
                print 'err % 8.1f'%min_mean_dist
            if min_mean_dist <= 10.0:
                if DETECT_SND is not None:
                    now = time.time()
                    if (now - self.last_sound_time) > 1.0:
                        DETECT_SND.Play()
                        self.last_sound_time = now
                if self.current_page == 'preview':
                    r=self.main_brain.reconstructor
                    for cam_id in self.cameras.keys():
                        pt,ln=r.find2d(cam_id,data3d,
                                       Lcoords=line3d,distorted=True)
                        self.cam_image_canvas.set_reconstructed_points(cam_id,([pt],[ln]))
            
    def OnNewCamera(self, cam_id, scalar_control_info, fqdnport):
        print 'new camera: ',cam_id
        # bookkeeping
        self.cameras[cam_id] = {'scalar_control_info':scalar_control_info}

    def OnCollectingBackground(self, cam_id, enable):
        cam_id = self._get_cam_id_for_button(widget)
        self.main_brain.set_collecting_background( cam_id, enable )

    def OnStopAllCollectingBg(self):
        pass

    def OnTakeBackground(self, cam_id):
        self.main_brain.take_background(cam_id)

    def OnClearBackground(self, cam_id):
        self.main_brain.clear_background(cam_id)

    def OnFindRCenter(self, cam_id):
        self.main_brain.find_r_center(cam_id)

    def OnCloseCamera(self, cam_id):
        self.main_brain.close_camera(cam_id) # eventually calls OnOldCamera
    
    def OnSetCameraThreshold(self, cam_id, value):
        value = float(value)
        self.main_brain.send_set_camera_property(cam_id,'diff_threshold',value)

    def OnSetCameraClearThreshold(self, cam_id, value):
        value = float(value)
        self.main_brain.send_set_camera_property(cam_id,'clear_threshold',value)

    def OnSetMaxFramerate(self, value):
        value = float(value)
        self.main_brain.send_set_camera_property(cam_id,'max_framerate',value)

    def OnArenaControl(self, cam_id, enable):
        self.main_brain.set_use_arena( cam_id, enable )

    def OnExtTrig(self, cam_id, enable):
        self.main_brain.send_set_camera_property( cam_id, 'trigger_source', widget.IsChecked() )

    def OnROI2(self, cam_id, enable):
        self.main_brain.send_set_camera_property( cam_id, 'roi2', widget.IsChecked() )

    def OnOldCamera(self, cam_id):
        sys.stdout.flush()
        self.OnRecordRawStop(warn=False)
        del self.cameras[cam_id]
    
def main():
    usage = '%prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option("--server", dest="server", type='string',
                      help="hostname of mainbrain SERVER",
                      default='',
                      metavar="SERVER")
    parser.add_option("--save-profiling-data",
                      default=False, action="store_true",
                      help="save data to profile/debug the Kalman-filter based tracker (WARNING: SLOW)",
                      )
    parser.add_option("--disable-sync-errors", dest='show_sync_errors',
                      default=True, action="store_false",
                      )
    (options, args) = parser.parse_args()

    rospy.init_node('flydra_mainbrain')

    # initialize GUI
    app = MainBrainApp(0)
    # create main_brain server (not started yet)
    main_brain = MainBrain.MainBrain(server=options.server,
                                     save_profiling_data=options.save_profiling_data,
                                     show_sync_errors=options.show_sync_errors,
                                     publish_ros=True)

    try:
        # connect server to GUI
        app.attach_and_start_main_brain(main_brain)
        rospy.spin()
        print 'mainloop over'
        del app
        
    finally:
        # stop main_brain server
        main_brain.quit()
        print '2nd(?) call to quit?'
    
if __name__ == '__main__':
    main()
