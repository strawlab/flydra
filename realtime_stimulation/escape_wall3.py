#!/usr/bin/env python
# $Id: calibrate_viewport.py 561 2005-06-30 23:22:01Z astraw $
from __future__ import division
import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import *
import pygame
from pygame.locals import *
from VisionEgg.Gratings import *
from VisionEgg.Textures import *
from VisionEgg.Text import Text
#from VisionEgg.MoreStimuli import Rectangle3D
import Numeric as nx
import math, socket, struct, select, sys

tf_hz_all = [0.0, 1.0, 5.0]
tf_hz_idx = 0

screen = Screen(
    fullscreen=True,#False,
    frameless=True,
    sync_swap=False, # vertical sync
    size=(1024,768),
    )
screen.parameters.bgcolor = (0.0,0.0,0.0) # black (RGB)

# coordinates for tunnel in world coordinates
rightwall = nx.array([[480,0,0],
                      [480,0,304],
                      [980,0,304],
                      [980,0,0]])
leftwall = rightwall + nx.array([0,305,0])

trigger_x = 730
trigger_radius = 25 # mm

downwind_left_wall = leftwall.copy()
downwind_left_wall[2,0] = trigger_x
downwind_left_wall[3,0] = trigger_x

upwind_left_wall = leftwall.copy()
upwind_left_wall[0,0] = trigger_x
upwind_left_wall[1,0] = trigger_x

downwind_wall_width = downwind_left_wall[2][0] - downwind_left_wall[0][0]
upwind_wall_width = upwind_left_wall[2][0] - upwind_left_wall[0][0]

wall_height = leftwall[2][2] - leftwall[0][2]

horiz_wavelength = 100 # mm
horiz_sf = 1/horiz_wavelength
tf = 0
contrast = 10000

grating_downwind = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = horiz_sf,
    temporal_freq_hz = tf,
    contrast = contrast,
    size = (downwind_wall_width, wall_height),
    lowerleft = downwind_left_wall[2],
    upperleft = downwind_left_wall[3],
    upperright = downwind_left_wall[0],
    lowerright = downwind_left_wall[1],
    max_alpha = 1.0,
    color1 = (0,1,0),
    ignore_time = True,
    phase_at_t0 = 90.0,
    )

grating_upwind = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = horiz_sf,
    temporal_freq_hz = tf,
    contrast = contrast,
    size = (upwind_wall_width, wall_height),
    lowerleft = upwind_left_wall[0],
    upperleft = upwind_left_wall[1],
    upperright = upwind_left_wall[2],
    lowerright = upwind_left_wall[3],
    max_alpha = 1.0,
    color1 = (0,1,0),
    ignore_time = True,
    phase_at_t0 = 90.0,
    )

# make sure the viewport parameters are the same in your other code
viewport = Viewport(screen=screen,
                    position=(0,0),
                    anchor='lowerleft',
                    size=(1024,768))


v1o = nx.array((-5,305,-5))
v2o = nx.array(( 5,305,-5))
v3o = nx.array(( 5,305, 5))
v4o = nx.array((-5,305, 5))
def fly_pos_to_verts(fly_xyz):
    X = nx.array(fly_xyz)
    X[1]=0
    return v1o+X, v2o+X, v3o+X, v4o+X

# calibration data (20050706)

fov_x = 60.1117038883
aspect_ratio = 1.33333333333
eye = (766.80000000000155, 802.0, 179.09999999999914)
center = (752.000000000005, 305.0, 182.09999999999928)
up = (-0.0040000000000000001, 0.0, 1.0)

projection = SimplePerspectiveProjection(fov_x=fov_x,aspect_ratio=aspect_ratio)
projection.look_at( eye, center, up )

viewport.set(projection=projection)
viewport.set(stimuli=[grating_downwind, grating_upwind])


#################################
class NetChecker:
    def __init__(self):
        self.data = ''
        self.x = None
        self.y = None
        self.z = None
        self.corrected_framenumber = None

        hostname = ''
        self.sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sockobj.setblocking(0)

        port = 28931

        self.sockobj.bind(( hostname, port))
        
    def get_last_xyz_fno(self):
        return (self.x, self.y, self.z), self.corrected_framenumber
    
    def check_network(self):
        fmt = '<iBfffffffffdf' # little endian
        fmt_size = struct.calcsize(fmt)

        # return if data not ready
        while 1:
            in_ready, trash1, trash2 = select.select( [self.sockobj], [], [], 0.0 )
            if not len(in_ready):
                break

            newdata, addr = self.sockobj.recvfrom(4096)

            mytime = time.time()
            self.data = self.data + newdata
            while len(self.data) >= fmt_size:
                tmp = struct.unpack(fmt,self.data[:fmt_size])
                self.data = self.data[fmt_size:]
                repr_error = tmp[-1]
                if repr_error < 10.0:
                    self.corrected_framenumber,line3d_valid,self.x,self.y,self.z = tmp[:5]
                    find3d_time = tmp[-2]
    ##            approx_latency = (mytime-find3d_time)*1000.0
    ##            print 'approx_latency',(approx_latency-min_latency) # display only varying part of latency
    ##            if approx_latency < min_latency:
    ##                min_latency = approx_latency
    ##            x = (xo - 1250) * 4 + 320
    ##            y = (yo - 150) * 4 + 240


#################################

def query_fly_in_trigger_volume(xyz):
    if xyz[0] is None:
        return False
    dist = abs(xyz[0]-trigger_x) # only consider x component
    if dist <= trigger_radius:
        return True
    else:
        return False

stim_dur_sec = 1.0
pause_dur_sec = 1.0

log_file = open(time.strftime( 'escape_wall%Y%m%d_%H%M%S.log' ), mode='wb')
#log_file = sys.stdout
log_file.write( '#trigger_x = %s\n'%str(trigger_x))
log_file.write( '#trigger_radius = %s\n'%str(trigger_radius))
log_file.write( '#stim_dur_sec = %s\n'%str(stim_dur_sec) )
log_file.write( '#pause_dur_sec = %s\n'%str(pause_dur_sec) )
log_file.write( '## trig_frame trig_time tf_hz\n')

nc = NetChecker()
frame_timer = FrameTimer()
quit_now = False
status = 'armed'
mode_start_time = VisionEgg.time_func()
trigger_corrected_framenumber = None
while not quit_now:
    
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key in (pygame.locals.K_q, pygame.locals.K_ESCAPE):
                quit_now = 1
        elif event.type in (QUIT,MOUSEBUTTONDOWN):
            quit_now = 1

    ######################

    nc.check_network()
    xyz, cf = nc.get_last_xyz_fno()

    if cf is not None:
        # do something useful
        1+1

    now = VisionEgg.time_func()
    
    if (status == 'armed' and trigger_corrected_framenumber!=cf and
        query_fly_in_trigger_volume(xyz)) :
        status = 'triggered'
        timetime = time.time()
        tf_hz_idx += 1
        tf_hz_idx = tf_hz_idx % len(tf_hz_all)
        tf_hz = tf_hz_all[tf_hz_idx]
        log_file.write('%d %s %f\n'%(cf,repr(timetime),tf_hz))
        mode_start_time = now
        trigger_corrected_framenumber = cf
    elif status == 'triggered':
        if (now-mode_start_time) >= stim_dur_sec:
            status = 'waiting'
            mode_start_time = now
        else:
            IFI = last_frame-now
            phase_change = tf_hz*360.0*IFI
            phase_downwind = grating_downwind.parameters.phase_at_t0 + phase_change
            phase_upwind = grating_upwind.parameters.phase_at_t0 + phase_change
            # move gratings
            grating_downwind.set(phase_at_t0 = phase_downwind)
            grating_upwind.set(phase_at_t0 = phase_upwind)
    elif status == 'waiting':
        if (now-mode_start_time) >= pause_dur_sec:
            status = 'armed'
    last_frame = now

    ######################
    
    screen.clear()
    viewport.draw()
    swap_buffers()
    frame_timer.tick()
frame_timer.log_histogram()
