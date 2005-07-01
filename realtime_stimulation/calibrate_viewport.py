#!/usr/bin/env python
# $Id$
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
import Numeric as nx
import math

screen = Screen(
    fullscreen=False,
    frameless=True,
    sync_swap=False, # vertical sync
    size=(1024,768),
    )
screen.parameters.bgcolor = (0.0,0.0,0.0) # black (RGB)

# coordinates for tunnel in world coordinates
rightwall = nx.array([[1000,0,0],
                      [1000,0,304],
                      [1500,0,304],
                      [1500,0,0]])
leftwall = rightwall + nx.array([0,305,0])

texture = Texture('grid-shrunk.gif')

wall_x = (leftwall[2][0]-leftwall[0][0])
wall_z = (leftwall[2][2]-leftwall[0][2])

print 'wall_x (%d) /wall_z (%d) = %f'%(wall_x,wall_z,wall_x/wall_z)
print 'texture.size = %s = %f'%(str(texture.size),texture.size[0]/texture.size[1])
print '-='*20
print 'IMPORTANT: Please make sure the above ratios match'
print '-='*20


# which wall are we calibrating?
wall = leftwall
# what initial guess do we use for our projector postion?
eye_start_offset = (0, 500, 0) # relative to center of wall
# make sure the viewport parameters are the same in your other code
viewport = Viewport(screen=screen,
                    position=(0,0),
                    anchor='lowerleft',
                    size=(1024,768))

stim = TextureStimulus3D(
    texture = texture,
    lowerleft = wall[0],
    upperleft = wall[1],
    upperright = wall[2],
    lowerright = wall[3],
    )

### starting guess
##projection = SimplePerspectiveProjection(fov_x=90.0)
##center = ((wall[0][0] + wall[2][0])/2,
##          (wall[0][1] + wall[2][1])/2,
##          (wall[0][2] + wall[1][2])/2,
##          )
##eye = nx.array(center) + nx.array(eye_start_offset)
##up = (0,0,1)
##projection.look_at( eye, center, up )

##viewport.set(projection=projection)
viewport.set(stimuli=[stim])

####################

status_text = Text()
viewport_overlay = Viewport(screen=screen,
                            position=(200,300),
                            anchor='lowerleft',
                            size=screen.size,
                            stimuli=[status_text])
status_var_main = 'None'
status_var_sub = 'scalar'
status_var_value = None

#####################

vec_x = nx.array((1,0,0))
vec_y = nx.array((0,1,0))
vec_z = nx.array((0,0,1))
def get_vec( sub ):
    if sub == 'x':
        return vec_x
    elif sub == 'y':
        return vec_y
    elif sub == 'z':
        return vec_z
    else:
        raise ValueError('did not understand %s'%sub)

frame_timer = FrameTimer()

# starting guess (from previous iteration)
fov_x = 58.283267922
aspect_ratio = 1.33333333333
eye = (1250.0, 805.0, 191.19999999999777)
center = (1250.0, 305.0, 195.59999999999752)
up = (0.028000000000000018, 0.0, 1.0)

quit_now = 0
pressing_up = 0
pressing_down = 0
pressing_shift = 0

while not quit_now:
    
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == pygame.locals.K_f:
                status_var_main = 'fov_x'
            elif event.key == pygame.locals.K_a:
                status_var_main = 'aspect_ratio'
            elif event.key == pygame.locals.K_e:
                status_var_main = 'eye'
            elif event.key == pygame.locals.K_c:
                status_var_main = 'center'
            elif event.key == pygame.locals.K_u:
                status_var_main = 'up'
            elif event.key == pygame.locals.K_s:
                status_var_sub = 'scalar'
            elif event.key == pygame.locals.K_x:
                status_var_sub = 'x'
            elif event.key == pygame.locals.K_y:
                status_var_sub = 'y'
            elif event.key == pygame.locals.K_z:
                status_var_sub = 'z'
            elif event.key == pygame.locals.K_UP:
                pressing_up = 1
            elif event.key == pygame.locals.K_DOWN:
                pressing_down = 1
            elif event.key in (pygame.locals.K_LSHIFT,pygame.locals.K_RSHIFT):
                pressing_shift = 1
            elif event.key in (pygame.locals.K_q, pygame.locals.K_ESCAPE):
                quit_now = 1
        elif event.type == KEYUP:
            if event.key == pygame.locals.K_UP:
                pressing_up = 0
            elif event.key == pygame.locals.K_DOWN:
                pressing_down = 0
            elif event.key in (pygame.locals.K_LSHIFT,pygame.locals.K_RSHIFT):
                pressing_shift = 0
        elif event.type in (QUIT,MOUSEBUTTONDOWN):
            quit_now = 1

    ######################

    if pressing_up:
        change = 0.001
    elif pressing_down:
        change = -0.001
    else:
        change = 0

    if pressing_shift:
        change = change*100

    #######################
    
    if status_var_main == 'fov_x':
        status_var_sub = 'scalar'
        fov_x = fov_x + change*fov_x
        status_var_value = fov_x
    elif status_var_main == 'aspect_ratio':
        status_var_sub = 'scalar'
        aspect_ratio = aspect_ratio + change*aspect_ratio
        status_var_value = aspect_ratio
    elif status_var_main in ('eye','center','up'):
        if status_var_sub == 'scalar': status_var_sub = 'x'
        if status_var_main == 'eye':
            eye = nx.array(eye) + get_vec( status_var_sub )*change*100
            tmp = eye*get_vec( status_var_sub )
            status_var_value = tmp[0] + tmp[1] + tmp[2]
        elif status_var_main == 'center':
            center = nx.array(center) + get_vec( status_var_sub )*change*100
            tmp = center*get_vec( status_var_sub )
            status_var_value = tmp[0] + tmp[1] + tmp[2]
        elif status_var_main == 'up':
            vec = get_vec( status_var_sub )
            up = nx.array(up) + vec*change*1
            tmp = up*get_vec( status_var_sub )
            status_var_value = tmp[0] + tmp[1] + tmp[2]            

    projection = SimplePerspectiveProjection(fov_x=fov_x,
                                             aspect_ratio=aspect_ratio)
    projection.look_at( eye, center, up )

    viewport.parameters.projection = projection

    status_text.parameters.text = ' '.join([status_var_main,status_var_sub,str(status_var_value)])

    ######################

    screen.clear()
    viewport.draw()
    viewport_overlay.draw()
    swap_buffers()
    frame_timer.tick()
frame_timer.log_histogram()

print 'fov_x =',fov_x
print 'aspect_ratio =',aspect_ratio
print 'eye =',tuple(eye)
print 'center =',tuple(center)
print 'up =',tuple(up)
