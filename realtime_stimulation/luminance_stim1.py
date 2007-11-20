#!/usr/bin/env python
from VisionEgg import *
start_default_logging(); watch_exceptions()

from VisionEgg.Core import *
from VisionEgg.Textures import *

import Image, ImageDraw # Python Imaging Library (PIL)

import OpenGL.GL as gl # PyOpenGL
import pygame
from pygame.locals import *

import numpy
if 1:
    import Pyro.core, Pyro.errors
    Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!
    Pyro.config.PYRO_TRACELEVEL = 3
    Pyro.config.PYRO_USER_TRACELEVEL = 3
    Pyro.config.PYRO_DETAILED_TRACEBACK = 1
    Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1
    ConnectionClosedError = Pyro.errors.ConnectionClosedError

if 1:
    #sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #mainbrain = ('192.168.10.42')
    #sender.sendto('x',(MainBrain.hostname,common_variables.trigger_network_socket_port))

    Pyro.core.initClient(banner=0)
    port = 9833
    name = 'main_brain'
    main_brain_hostname = '192.168.10.42'
    main_brain_URI = "PYROLOC://%s:%d/%s" % (main_brain_hostname,port,name)
    print 'connecting to',main_brain_URI
    main_brain = Pyro.core.getProxyForURI(main_brain_URI)
    #main_brain._setOneway(['log_message'])

    main_brain.log_message('stimulus', time.time(), 'starting' ) # check that we can connect


scale = 20 # magification of checkerboard (this many pixels per texture element)

screen = get_default_screen()

dynamic_checkerboard_size = (screen.size[0]//scale,screen.size[1]//scale) # width, height in texture elements

# allocate temporary texture in grayscale mode for dynamic texture
temp_grayscale_image = Image.new("L",dynamic_checkerboard_size,0)
temp_texture = Texture(temp_grayscale_image)
    
# create TextureStimulus for dynamic stimulus
scaled_dynamic_size = (scale*dynamic_checkerboard_size[0],scale*dynamic_checkerboard_size[1])

# find center of screen
x = screen.size[0]/2.0
y = screen.size[1]/2.0
dynamic_checkerboard = TextureStimulus(texture=temp_texture,
                                       position=(x,y),
                                       anchor="center",
                                       mipmaps_enabled=0,
                                       size=scaled_dynamic_size,
                                       texture_min_filter=gl.GL_NEAREST,
                                       texture_mag_filter=gl.GL_NEAREST,
                                       )

viewport = Viewport(screen=screen,
                    stimuli=[
                             dynamic_checkerboard]
                    )

dynamic_texture_object = dynamic_checkerboard.parameters.texture.get_texture_object()
#width,height = dynamic_checkerboard_size

quit_now = 0
contrast = 1.0

# zeros and ones
#d = numpy.random.randint(0,2,size=(dynamic_checkerboard_size[1],dynamic_checkerboard_size[0]))
checkerboard = numpy.zeros((dynamic_checkerboard_size[1],dynamic_checkerboard_size[0]))
for row in range(dynamic_checkerboard_size[1]):
    for col in range(dynamic_checkerboard_size[0]):
        if row%2:
            if col%2:
                checkerboard[row,col] = 1
        else:
            if not col%2:
                checkerboard[row,col] = 1

lums = numpy.logspace(numpy.log10(0.05),numpy.log10(.5),6)
lums = lums[::-1] # reverse
lum_dur = 5*60 # 5 minutes
#lum_dur = 1
print 'lums',lums

tstart = time.time()
last_lum = None

while not quit_now:
    for event in pygame.event.get():
        if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
            quit_now = 1

    # update the image
    tnow = time.time()
    tdiff = tnow-tstart
    #print 'tdiff',tdiff
    lum_idx_unwrapped = math.floor(tdiff/lum_dur)
    #print 'lum_idx_unwrapped',lum_idx_unwrapped
    lum_idx = lum_idx_unwrapped%len(lums)
    lum_idx = int(lum_idx)
    normalized_mean_luminance = lums[lum_idx]
    if normalized_mean_luminance != last_lum:
        msg = 'luminance %f, contrast %f'%(normalized_mean_luminance,contrast)
        print msg
        last_lum = normalized_mean_luminance
        main_brain.log_message('stimulus', time.time(), msg )

        #sender.sendto('luminace %')

    #normalized_mean_luminance = (tnow-tstart)%0.5

    mean_luminance = normalized_mean_luminance*255.0
    contrast_scale = contrast*mean_luminance
    #print 'contrast_scale',contrast_scale
    Lmax = mean_luminance + contrast_scale
    Lmin = mean_luminance - contrast_scale
    Ldiff = Lmax-Lmin
    #print 'Ldiff',Ldiff

    scaled_data = (checkerboard*Ldiff + Lmin)
    #print 'normalized_mean_luminance, contrast, Lmin, Lmax,',normalized_mean_luminance, contrast, scaled_data.min(), scaled_data.max()
    if scaled_data.max() > 255.0:
        raise ValueError("cannot produce this combination of luminance and contrast!")
    scaled_data = scaled_data.astype(numpy.uint8)
    #print 'normalized_mean_luminance, contrast, Lmin, Lmax,',normalized_mean_luminance, contrast, scaled_data.min(), scaled_data.max()
    dynamic_texture_object.put_sub_image( scaled_data )

    screen.clear()
    viewport.draw()
    swap_buffers()
