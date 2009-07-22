#!/usr/bin/env python
"""Load a texture with alpha from a file and draw with mask."""

import os
import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import *
from VisionEgg.Textures import *
import pygame
from pygame.locals import *

this_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
filename = os.path.join(this_dir,"spiral.png")
texture = Texture(filename)

screen = get_default_screen()
screen.set(bgcolor=(0,0.0,0)) # black background

mask = Mask2D(function='circle',     # also supports 'gaussian'
              radius_parameter=130,  # sigma for gaussian, radius for circle (units: num_samples)
              num_samples=(512,512)) # this many texture elements in mask (covers whole size specified below)


# Create the instance of TextureStimulus
stimulus = TextureStimulus(texture = texture,
                           mask = mask,
                           position = (screen.size[0]/2.0,screen.size[1]/2.0),
                           anchor = 'center',
                           shrink_texture_ok=1,
                           internal_format=gl.GL_RGBA,
                           size=(1024,1024),
                           )

viewport = Viewport(screen=screen,
                    stimuli=[stimulus])

frame_timer = FrameTimer()
quit_now = False
spinning = True
while not quit_now and spinning:
    for event in pygame.event.get():
        if event.type == QUIT:
            quit_now = True
        elif event.type in (KEYDOWN,MOUSEBUTTONDOWN):
            spinning = False
    screen.clear()
    stimulus.parameters.angle = (VisionEgg.time_func()*-360.0)%360.0 # rotate
    viewport.draw()
    swap_buffers()
    frame_timer.tick()
while not quit_now:
    for event in pygame.event.get():
        if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
            quit_now = True
    screen.clear()
    viewport.draw()
    swap_buffers()
    frame_timer.tick()
frame_timer.log_histogram()
