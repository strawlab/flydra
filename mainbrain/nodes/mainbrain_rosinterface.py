#!/usr/bin/env python
from __future__ import division
import roslib; roslib.load_manifest('mainbrain')
import rospy
import socket
import Pyro.core
import pickle
from mainbrain.srv import *
import flydra.common_variables

import threading
import struct


#LOGLEVEL = rospy.DEBUG
#LOGLEVEL = rospy.INFO
LOGLEVEL = rospy.WARN
#LOGLEVEL = rospy.ERROR
#LOGLEVEL = rospy.FATAL



# Thread to echo timestamps from mainbrain, to camera, back to mainbrain.
def ThreadEchoTimestamp(iCamera, camera):
    # Create timestamp sending socket.
    socketSendTimestamp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    portSendTimestamp = rospy.get_param('mainbrain/port_timestamp', 28993)

    # Offer a receiving socket for echo_timestamp from mainbrain:  localhost:28995,6,7,8,...
    socketReceiveTimestamp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = ''
    portReceiveTimestampBase = rospy.get_param('mainbrain/port_timestamp_camera_base', 28995)
    portReceiveTimestamp = portReceiveTimestampBase + iCamera
    try:
        socketReceiveTimestamp.bind(( hostname, portReceiveTimestamp))
    except socket.error, err:
        if err.args[0]==98:
            rospy.logwarn('EchoTimestamp for camera %d not available because port %d in use' % (iCamera, portReceiveTimestamp))


    
    fmt = rospy.get_param('mainbrain/timestamp_echo_fmt1', '<d') #flydra.common_variables.timestamp_echo_fmt_diff
    
    while True:
        # Receive timestamp from mainbrain.
        try:
            packTimestamp, (orig_host,orig_port) = socketReceiveTimestamp.recvfrom(4096)
        except socket.error, err:
            if err.args[0] == errno.EINTR: # interrupted system call
                continue
            raise

        if struct is None: # this line prevents bizarre interpreter shutdown errors
            return


        # Send timestamp to camera & back.
        timeMainbrain = struct.unpack(fmt,packTimestamp)[0]
        timeCamera = camera['echo_timestamp'](time=timeMainbrain)
        
        # Send both times back to mainbrain.
        packTimestamp2 = packTimestamp + struct.pack( fmt, timeCamera.time)
        socketSendTimestamp.sendto(packTimestamp2, (orig_host, portSendTimestamp))
        
    

class MainbrainRosInterface(object):
    """Provide a ROS interface to mainbrain (using services)."""
    def __init__(self):
        rospy.init_node('MainbrainRosInterface', log_level=LOGLEVEL)
        Pyro.core.initClient(banner=0)

        self.cameras = []
        
         

        ##################################################################
        # Connections to Mainbrain.
        ##################################################################

        # Construct a URI to mainbrain.
        portMainbrain = rospy.get_param('port_mainbrain', 9833)
        nameMainbrain = 'main_brain'
        try:
            self.hostnameMainbrain = socket.gethostbyname(rospy.get_param('mainbrain/hostname', 'brain1'))
        except:
            try:
                self.hostnameMainbrain = socket.gethostbyname(socket.gethostname()) # try localhost
            except: #socket.gaierror?
                self.hostnameMainbrain = ''
        uriMainbrain = "PYROLOC://%s:%d/%s" % (self.hostnameMainbrain, portMainbrain, nameMainbrain)


        # Connect to mainbrain.
        print 'Connecting to ',uriMainbrain
        try:
            self.proxyMainbrain = Pyro.core.getProxyForURI(uriMainbrain)
        except:
            print 'ERROR connecting to',uriMainbrain
            raise
        print "Connected."
        self.proxyMainbrain._setOneway(['set_image','set_fps','close','log_message','receive_missing_data'])



        ##################################################################
        # Connections to Camnode.
        ##################################################################

    

        # TriggerRecording service.
        self.timeRecord = rospy.Time.now().to_sec()
        self.isRecording = False
        hostnameLocal = ''
        portTriggerRecording = 30043 # arbitrary number
        self.socketTriggerRecording = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            self.socketTriggerRecording.bind((hostnameLocal, portTriggerRecording))
            print 'Created udp server on port ', portTriggerRecording
        except socket.error, err:
            if err.args[0]==98: # port in use
                rospy.logwarn('Port %s:%s in use.  Cannot toggle recording state.' % (hostnameLocal, portTriggerRecording))
                self.socketTriggerRecording = None

        if self.socketTriggerRecording is not None:
            self.socketTriggerRecording.setblocking(0)



        # Expose the flydra_mainbrain API functions as ROS services.  Same as the Pyro function call, except with a few parameters pickled.
        rospy.Service ('mainbrain/get_version',              SrvGetVersion, self.callback_get_version)
        rospy.Service ('mainbrain/register_camera',          SrvRegisterCamera, self.callback_register_camera)
        rospy.Service ('mainbrain/get_cam2mainbrain_port',   SrvGetCam2MainbrainPort, self.callback_get_mainbrain_port)
        rospy.Service ('mainbrain/get_and_clear_commands',   SrvGetAndClearCommands, self.callback_get_and_clear_commands)
        rospy.Service ('mainbrain/set_fps',                  SrvSetFps, self.callback_set_fps)
        rospy.Service ('mainbrain/set_image',                SrvSetImage, self.callback_set_image)
        rospy.Service ('mainbrain/log_message',              SrvLogMessage, self.callback_log_message)
        rospy.Service ('mainbrain/receive_missing_data',     SrvReceiveMissingData, self.callback_receive_missing_data)
        rospy.Service ('mainbrain/close_camera',             SrvClose, self.callback_close_camera)
        rospy.Service ('mainbrain/get_recording_status',     SrvGetRecordingStatus, self.callback_get_recording_status)


    def ICameraFromId (self, id):
        for iCamera in range(len(self.cameras)):
            if self.cameras[iCamera]['id']==id:
                break
            
        return iCamera
        
        

    ###########################################################################
    # Callbacks    
    ###########################################################################

    def callback_get_version (self, srvreqGetVersion):
        #srvrespGetVersion = SrvGetVersionResponse()
        versionMainbrain = self.proxyMainbrain.get_version()
        return {'version': versionMainbrain}

        
    def callback_register_camera (self, srvreqRegisterCamera):
        # Add a dict for the camera.
        iCamera = len(self.cameras)
        self.cameras.append({})
        
        force_cam_id = srvreqRegisterCamera.force_cam_id
        try:
            if len(force_cam_id)==0:
                force_cam_id = None
        except TypeError:
            force_cam_id = None
                    

        # Register the camera with mainbrain.        
        port = rospy.get_param('mainbrain/port_camera_base', 9834) + iCamera
        scalar_control_info = pickle.loads(srvreqRegisterCamera.pickled_scalar_control_info)
        idCamera = self.proxyMainbrain.register_new_camera(cam_no=srvreqRegisterCamera.cam_no, 
                                                           scalar_control_info=scalar_control_info, 
                                                           port=port, #srvreqRegisterCamera.port,  # Mainbrain now talks with us, not the actual camnode.
                                                           force_cam_id = force_cam_id)

        print "Registering camera %d as %s" % (iCamera, idCamera)
        
        # Keep the camera's ID.
        self.cameras[iCamera]['id'] = idCamera


        # Connect to the camera's echo_timestamp ROS service, e.g. "echo_timestamp_camera1".
        stSrv = idCamera+'/echo_timestamp'
        rospy.wait_for_service(stSrv)
        self.cameras[iCamera]['echo_timestamp'] = rospy.ServiceProxy(stSrv, SrvEchoTimestamp)
        print 'Connected to service %s' % stSrv

        # Connect to mainbrain's coordinates port.
        self.portMainbrainCoordinates = self.proxyMainbrain.get_cam2mainbrain_port(idCamera)
        #self.portMainbrainCoordinates = rv.port

        if rospy.get_param('mainbrain/network_protocol','udp') == 'udp':
            self.cameras[iCamera]['socketCoordinates'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif rospy.get_param('mainbrain/network_protocol','udp') == 'tcp':
            self.cameras[iCamera]['socketCoordinates'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cameras[iCamera]['socketCoordinates'].connect((self.hostnameMainbrain, self.portMainbrainCoordinates))
        else:
            raise ValueError('unknown network_protocol')


        # Offer the coordinate data collection service, e.g. "coordinates_camera1".
        rospy.Service ('mainbrain/coordinates/'+idCamera, 
                       SrvCoordinates, 
                       self.callback_coordinates)


        # Launch a thread to handle echo_timestamp.
        print 'Starting thread %s...' % ('thread_timestamp_'+idCamera)
        self.cameras[iCamera]['thread_timestamp'] = threading.Thread(target=ThreadEchoTimestamp, name='thread_timestamp_'+idCamera, args=(iCamera,self.cameras[iCamera],))
        self.cameras[iCamera]['thread_timestamp'].setDaemon(True) # quit that thread if it's the only one left...
        self.cameras[iCamera]['thread_timestamp'].start()
        print 'Started thread %s' % ('thread_timestamp_'+idCamera)

        
        return {'cam_id': idCamera}


    def callback_get_mainbrain_port (self, srvreqGetCam2MainbrainPort):
        portMainbrain = self.proxyMainbrain.get_cam2mainbrain_port(srvreqGetCam2MainbrainPort.cam_id)
        return {'port': portMainbrain}


    def callback_get_and_clear_commands (self, srvreqGetAndClearCommands):
        cmds=self.proxyMainbrain.get_and_clear_commands(srvreqGetAndClearCommands.cam_id)
        return {'pickled_cmds': pickle.dumps(cmds)}
    
    
    def callback_set_fps (self, srvreqSetFps):
        self.proxyMainbrain.set_fps(srvreqSetFps.cam_id, 
                                    srvreqSetFps.fps)
        return {}


    def callback_set_image (self, srvreqSetImage):
        coord_and_image = pickle.loads(srvreqSetImage.pickled_coord_and_image)
        self.proxyMainbrain.set_image(srvreqSetImage.cam_id, 
                                      coord_and_image)
        return {}


    def callback_log_message (self, srvreqLogMessage):
        self.proxyMainbrain.log_message(srvreqLogMessage.cam_id, 
                                        srvreqLogMessage.host_timestamp, 
                                        srvreqLogMessage.message)
        return {}

    def callback_receive_missing_data (self, srvreqReceiveMissingData):
        missing_data = pickle.loads(srvreqReceiveMissingData.pickled_missing_data)
        self.proxyMainbrain.receive_missing_data(srvreqReceiveMissingData.cam_id, 
                                                 srvreqReceiveMissingData.framenumber_offset, 
                                                 missing_data)
        return {}


    def callback_close_camera (self, srvreqClose):
        self.proxyMainbrain.close(srvreqClose.cam_id)
        return {}
    

    # Not sure that mainbrain uses this anymore.
    def callback_get_recording_status (self, srvreqGetRecordingStatus):        
        msg = None
        if self.socketTriggerRecording is not None:
            try:
                msg, addr = self.socketTriggerRecording.recvfrom(4096) # Call mainbrain to get any trigger recording commands.
            except socket.error, err:
                if err.args[0] == 11: #Resource temporarily unavailable
                    pass

        #print ">>>", msg, "<<<", self.isRecording 
        
        if msg=='record_ufmf':
            if self.isRecording==False:
                self.isRecording = True
                print 'Start saving video.'
        
        elif msg==None:
            if (self.isRecording==True) and (rospy.Time.now().to_sec() - self.timeRecord >= 4): # Record at least 4 secs of video.
                self.isRecording = False
                print 'Stop saving video.'

        self.timeRecord = rospy.Time.now().to_sec()
                

        return {'status': self.isRecording}


        
    def callback_coordinates(self, srvreqCoordinates):
        iCamera = self.ICameraFromId(srvreqCoordinates.id)
        socketCoordinates = self.cameras[iCamera]['socketCoordinates']
        
        if rospy.get_param('mainbrain/network_protocol','udp') == 'udp':
            try:
                socketCoordinates.sendto(srvreqCoordinates.data, (self.hostnameMainbrain, self.portMainbrainCoordinates))
            except socket.error, err:
                print >> sys.stderr, 'WARNING: ignoring error:'
                traceback.print_exc()

        elif rospy.get_param('mainbrain/network_protocol','udp') == 'tcp':
            socketCoordinates.send(srvreqCoordinates.data)
        else:
            raise ValueError('unknown network_protocol')

        return {}
    
    
    ###########################################################################
    # Main
    ###########################################################################
    
    def Main(self):
        rospy.spin()


if __name__=='__main__':
    mainbrainRosInterface = MainbrainRosInterface()
    mainbrainRosInterface.Main()


