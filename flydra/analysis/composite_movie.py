#!/usr/bin/env python
import os
import Image
import glob

opj = os.path.join
dir = '/home/astraw/mcsc_data2.good-cal'

cam1_name = opj(dir,'point.cam1.%d.jpg')
cam2_name = opj(dir,'point.cam2.%d.jpg')
cam3_name = opj(dir,'point.cam3.%d.jpg')
cam4_name = opj(dir,'point.cam4.%d.jpg')
rec1_name = 'im%04d.png'
##cam1 = glob.glob(opj(dir,'point.cam1.*.jpg'))
##cam2 = glob.glob(opj(dir,'point.cam2.*.jpg'))
##cam3 = glob.glob(opj(dir,'point.cam3.*.jpg'))
##cam4 = glob.glob(opj(dir,'point.cam4.*.jpg'))
##rec1 = glob.glob('im*.png')
rec2_names = glob.glob('rot*.png')

##assert len(cam1)==len(cam2)
##assert len(cam1)==len(cam3)
##assert len(cam1)==len(cam4)
##assert len(cam1)==len(rec)

for FRAME_NO in range(410,472):
    frame_size=800,480
    frame = Image.new("RGB",frame_size,(0,255,0))
    im_cam1 = Image.open(cam1_name%FRAME_NO)
    im_cam2 = Image.open(cam2_name%FRAME_NO)
    im_cam3 = Image.open(cam3_name%FRAME_NO)
    im_cam4 = Image.open(cam4_name%FRAME_NO)
    im_rec1 = Image.open(rec1_name%FRAME_NO)
##    im_cam2 = Image.open(cam2[FRAME_NO])
##    im_cam3 = Image.open(cam3[FRAME_NO])
##    im_cam4 = Image.open(cam4[FRAME_NO])
##    im_rec = Image.open(rec[FRAME_NO])

    cam_size = 318,238
    shrink_filter = Image.ANTIALIAS

    im_cam1=im_cam1.resize(cam_size,shrink_filter)
    im_cam2=im_cam2.resize(cam_size,shrink_filter)
    im_cam3=im_cam3.resize(cam_size,shrink_filter)
    im_cam4=im_cam4.resize(cam_size,shrink_filter)

    rec_size = 160,480
    im_rec1=im_rec1.resize(rec_size,shrink_filter)

    ims = [[im_cam1, im_cam2], [im_cam3, im_cam4]]
    for i in range(len(ims)):
        for j in range(len(ims[0])):
            im = ims[i][j]
            x1=1+j*(cam_size[0]+1)
            y1=1+i*(cam_size[1]+1)
            frame.paste( im, (x1,y1))

    x1,y1=(640,0)
    frame.paste( im_rec1, (x1,y1))
    fname = 'final%04d.png'%(FRAME_NO+1)
    print 'saving',fname,
    frame.save(fname)
    print 'done'

FRAME_NO+=1

for rec2_name in rec2_names:
    print 'loading',rec2_name
    im_rec2 = Image.open(rec2_name)
    im_rec2=im_rec2.resize(rec_size,shrink_filter)
    x1,y1=(640,0)
    frame.paste( im_rec2, (x1,y1))
    fname = 'final%04d.png'%(FRAME_NO+1)
    print 'saving',fname,
    frame.save(fname)
    print 'done'
    FRAME_NO+=1
