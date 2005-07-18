#!/usr/bin/env python
from __future__ import division
import os
print [(key, os.environ[key]) for key in os.environ if key.startswith('SDL')]

import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import *
import pygame
from pygame.locals import *
from VisionEgg.Gratings import *
from VisionEgg.Textures import *
from VisionEgg.Text import Text
from VisionEgg.MoreStimuli import Target2D
import Numeric as nx
import math

screen = get_default_screen()
screen.parameters.bgcolor = (0.0,0.0,0.0) # black (RGB)

# coordinates for tunnel
rightwall = nx.array([[0,0,0],
                      [0,0,300],
                      [1200,0,300],
                      [1200,0,0]])
leftwall = rightwall + nx.array([0,300,0])

draw_overview = True

# This uses images temporarily until I get sinusoidal gratings
# working in 3D. -- ADS

filename = os.path.join(VisionEgg.config.VISIONEGG_SYSTEM_DIR,"data","panorama.jpg")
texture = Texture(filename)

forwardbackward_grating_left = TextureStimulus3D(
    texture = texture,
    lowerleft = leftwall[0],
    upperleft = leftwall[1],
    upperright = leftwall[2],
    lowerright = leftwall[3],
    )
updown_grating_left = TextureStimulus3D(
    texture = texture,
    lowerleft = leftwall[0],
    upperleft = leftwall[1],
    upperright = leftwall[2],
    lowerright = leftwall[3],
    )

forwardbackward_grating_right = TextureStimulus3D(
    texture = texture,
    lowerleft = rightwall[0],
    upperleft = rightwall[1],
    upperright = rightwall[2],
    lowerright = rightwall[3],
    )
updown_grating_right = TextureStimulus3D(
    texture = texture,
    lowerleft = rightwall[0],
    upperleft = rightwall[1],
    upperright = rightwall[2],
    lowerright = rightwall[3],
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
    projection_overview = SimplePerspectiveProjection(fov_x=90.0)
    overview_eye = (60, -700, 500)
    overview_center = (600, 150, 150)
    overview_up = (0,0,1)
    projection_overview.look_at( overview_eye, overview_center, overview_up )

    viewport_overview = Viewport( screen=screen,
                                  position = (mid_x,mid_y),
                                  anchor='center',
                                  size=(screen.size[0]/4,
                                        screen.size[1]/4),
                                  projection = projection_overview,
                                  stimuli = [forwardbackward_grating_left,
                                             updown_grating_left,
                                             forwardbackward_grating_right,
                                             updown_grating_right],
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
                                              stimuli=[overview_rect] )
    viewport_overview_2d_overlay = Viewport( screen=screen,
                                             position = (mid_x,mid_y),
                                             anchor='center',
                                             size=(screen.size[0]/4,
                                                   screen.size[1]/4),
                                             stimuli=[overview_text] )

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
            
    screen.clear()
    viewport_left.draw()
    viewport_right.draw()
    if draw_overview:
        viewport_overview_2d_underlay.draw()
        viewport_overview.draw()
        viewport_overview_2d_overlay.draw()
    swap_buffers()
    frame_timer.tick()
frame_timer.log_histogram()
