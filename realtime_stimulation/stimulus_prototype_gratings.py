#!/usr/bin/env python
from __future__ import division
import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import *
import pygame
from pygame.locals import *
from VisionEgg.Gratings import *
from VisionEgg.Textures import *
from VisionEgg.Text import Text
from VisionEgg.MoreStimuli import Target2D
from VisionEgg.Textures import Mask2D
import Numeric as nx
import math

screen = get_default_screen()
screen.parameters.bgcolor = (0.0,0.0,0.0) # black (RGB)

# coordinates for tunnel (in mm)
rightwall = nx.array([[0,0,0],
                      [0,0,300],
                      [1200,0,300],
                      [1200,0,0]])
leftwall = rightwall + nx.array([0,300,0])

draw_overview = True

wall_width = leftwall[2][0] - leftwall[0][0]
wall_height = leftwall[2][2] - leftwall[0][2]

horiz_wavelength = 100 # mm
horiz_sf = 1/horiz_wavelength
vert_sf = horiz_sf # same for now
tf = 1.0

horiz_size = (wall_width, wall_height)
vert_size = (wall_height, wall_width)

vertical_alpha = 0.4

forwardbackward_grating_left = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = horiz_sf,
    temporal_freq_hz = tf,
    size = horiz_size,
    lowerleft = leftwall[2],
    upperleft = leftwall[3],
    upperright = leftwall[0],
    lowerright = leftwall[1],
    max_alpha = 1.0,
    )
updown_grating_left = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = vert_sf,
    temporal_freq_hz = tf,
    size = vert_size,
    lowerleft = leftwall[1],
    upperleft = leftwall[2],
    upperright = leftwall[3],
    lowerright = leftwall[0],
    max_alpha = vertical_alpha,
    )

forwardbackward_grating_right = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = horiz_sf,
    temporal_freq_hz = tf,
    size = horiz_size,
    lowerleft = rightwall[2],
    upperleft = rightwall[3],
    upperright = rightwall[0],
    lowerright = rightwall[1],
    max_alpha = 1.0,
    )
updown_grating_right = SinGrating3D(
    depth_test=False, # needed for overlay of other grating
    spatial_freq = vert_sf,
    temporal_freq_hz = tf,
    size = vert_size,
    lowerleft = rightwall[1],
    upperleft = rightwall[2],
    upperright = rightwall[3],
    lowerright = rightwall[0],
    max_alpha = vertical_alpha,
    )

projection_left = SimplePerspectiveProjection(fov_x=90.0)
left_eye = (600, 1000, 150)
left_center = (600, 300, 150)
left_up = (0,0,-1)
projection_left.look_at( left_eye, left_center, left_up )

projection_right = SimplePerspectiveProjection(fov_x=90.0)
right_eye = (600, -700, 150)
right_center = (600, 0, 150)
right_up = (0,0,1)
projection_right.look_at( right_eye, right_center, right_up )

mid_x = screen.size[0]/2
mid_y = screen.size[1]/2
viewport_left = Viewport(screen=screen,
                         position=(0,0),
                         anchor='lowerleft',
                         size=(screen.size[0],mid_y),
                         projection=projection_left,
                         stimuli=[forwardbackward_grating_left,
                                  updown_grating_left])
viewport_right = Viewport(screen=screen,
                          position=(0,mid_y),
                          anchor='lowerleft',
                          size=(screen.size[0],mid_y),
                          projection=projection_right,
                          stimuli=[forwardbackward_grating_right,
                                   updown_grating_right
                                   ])

if draw_overview:
    upwind_vertex = (rightwall[2] + leftwall[3])/2
    projection_overview = SimplePerspectiveProjection(fov_x=90.0)
    overview_eye = (60, -700, 500)
    overview_center = (600, 150, 150)
    overview_up = (0,0,1)
    projection_overview.look_at( overview_eye, overview_center, overview_up )

    upwind_text = Text( text = "upwind",
                        color    = (0.0, 0.9, 1.0),
                        anchor = "center",
                        font_size = 20,
                        )
    viewport_overview = Viewport( screen=screen,
                                  position = (mid_x,mid_y),
                                  anchor='center',
                                  size=(screen.size[0]/4,
                                        screen.size[1]/4),
                                  projection = projection_overview,
                                  stimuli = [forwardbackward_grating_left,
                                             updown_grating_left,
                                             forwardbackward_grating_right,
                                             updown_grating_right,
                                             ],
                                  )
    overview_text = Text( text     = "overview",
                          position = (0,0),
                          anchor   = 'lowerleft',
                          color    = (0.0, 0.9, 1.0),
                          )
    overview_rect = Target2D( color    = (0.0, 0.1, 0.2),
                              anchor = 'lowerleft',
                              position = (0,0),
                              size= (screen.size[0]/4,
                                     screen.size[1]/4),
                              )
    viewport_overview_2d_underlay = Viewport( screen=screen,
                                              position = (mid_x,mid_y),
                                              anchor='center',
                                              size=(screen.size[0]/4,
                                                    screen.size[1]/4),
                                              stimuli=[overview_rect],
                                              )
    viewport_overview_2d_overlay = Viewport( screen=screen,
                                             position = (mid_x,mid_y),
                                             anchor='center',
                                             size=(screen.size[0]/4,
                                                   screen.size[1]/4),
                                             stimuli=[overview_text] )

    viewport2D_whole_screen = Viewport(screen=screen,
                                       stimuli=[upwind_text],
                                       )

frame_timer = FrameTimer()
quit_now = 0
while not quit_now:
    for event in pygame.event.get():
        if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
            quit_now = 1

    if draw_overview:
        projection_overview = SimplePerspectiveProjection(fov_x=90.0)
        t = VisionEgg.time_func()
        overview_eye = (700*math.cos(0.1*t*2*math.pi)+overview_center[0],
                        700*math.sin(0.1*t*2*math.pi)+overview_center[1],
                        500+overview_center[2]) # orbit
        projection_overview.look_at( overview_eye, overview_center, overview_up )

        viewport_overview.parameters.projection = projection_overview
        upwind_vertex_window_coords = viewport_overview.eye_2_window(upwind_vertex)
        upwind_text.set(position = (upwind_vertex_window_coords[0], upwind_vertex_window_coords[1]))
            
    screen.clear()
    viewport_left.draw()
    viewport_right.draw()
    if draw_overview:
        viewport_overview_2d_underlay.draw()
        viewport_overview.draw()
        viewport_overview_2d_overlay.draw()
        viewport2D_whole_screen.draw()
    swap_buffers()
    frame_timer.tick()
frame_timer.log_histogram()
