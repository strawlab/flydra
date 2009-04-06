import pkg_resources
import pickle, os
import numpy as np
import benu
import motmot.FastImage.FastImage as FastImage
import motmot.realtime_image_analysis.realtime_image_analysis \
       as realtime_image_analysis
import ufmf_tools
import scipy.ndimage

if 1:
    fd = open('classifier_images.pkl',mode='r')
    classifier_images = pickle.load(fd)
    fd.close()

    fpc = realtime_image_analysis.FitParamsClass() # allocate FitParamsClass

    for count,ci in enumerate(classifier_images):
        (frame, ufmf_fname,
         y_slice, x_slice,
         ufmf_frame_timestamp,
         raw, # raw
         absdiff, # absdiff
         label, # label
         ) = ci
        fname = os.path.split(ufmf_fname)[-1]
        cam_id = ufmf_tools.get_cam_id_from_ufmf_fname(fname)

        if 0:
            print
            print frame
            print 'raw',raw
            print 'label',label

        show_RGB = np.empty( (raw.shape[0], raw.shape[1], 3), dtype=np.float32)
        show_RGB[:,:,0] = raw # R
        show_RGB[:,:,1] = raw # G
        show_RGB[:,:,2] = raw # B

        show_RGB[:,:,0] += label.astype(np.float32)*50 # highlight detected object
        show_RGB[:,:,1] += label.astype(np.float32)*50 # highlight detected object
        show_RGB[:,:,2] += label.astype(np.float32)*100 # highlight detected object

        show_RGB = np.clip( show_RGB, 0, 255)

        save_fname = 'classifier%05d.png'%(count+1,)
        save_res = 300
        w,h = show_RGB.shape[1], show_RGB.shape[0]
        aspect = float(w)/float(h)
        save_w = save_res
        save_h = int( save_res/aspect )
        canv = benu.Canvas(save_fname, save_w, save_h)
        display_rect = (0,0,save_w,save_h)
        user_rect = (x_slice.start,y_slice.start,w,h)

        with canv.set_user_coords(display_rect,user_rect):
            canv.imshow( show_RGB.astype(np.uint8), x_slice.start,y_slice.start )

            min_absdiff = 5
            for absdiff_max_frac_thresh,color_rgba in [
                (0.8, (0.5,1.0,0.5,0.5) ), # green
                (0.5, (0.8,0.8,0.5,0.5) ), # yellow
                (0.3, (1.0,0.5,0.5,0.5) ), # red
                ]:
                thresh_val = np.max(absdiff)*absdiff_max_frac_thresh
                thresh_val = max(min_absdiff,thresh_val)
                thresh_im = absdiff > thresh_val
                labeled_im,n_labels = scipy.ndimage.label(thresh_im)

                for i in range(n_labels):
                    this_label_im = labeled_im==i+1

                    if cam_id != 'cam6_1':
                        # pixel_aspect = 2
                        this_label_im = np.repeat(this_label_im,2,axis=0)
                    fast_foreground = FastImage.asfastimage(this_label_im.astype(np.uint8) )
                    (x0_roi, y0_roi, weighted_area, slope, eccentricity) = fpc.fit(fast_foreground)
                    if cam_id != 'cam6_1':
                        # pixel_aspect = 2
                        y0_roi *= 0.5

                    theta = np.arctan(slope)
                    r=10.0
                    dx = np.cos(theta)*r
                    dy = np.sin(theta)*r

                    canv.plot( [x_slice.start+x0_roi-dx, x_slice.start+x0_roi+dx],
                               [y_slice.start+y0_roi-dy, y_slice.start+y0_roi+dy],
                               color_rgba=color_rgba,
                               )
                    canv.scatter( [x_slice.start+x0_roi],
                                  [y_slice.start+y0_roi],
                                  color_rgba=color_rgba,
                                  )

        canv.text('%s %d %dx%d'%(cam_id,frame,w,h),
        ## canv.text('%s %d %dx%d: area=%.1f ecc=%.1f'%(cam_id,frame,w,h,
        ##                                              pixel_area, eccentricity),
                  5,save_h-5,
                  color_rgba=(.5,1,.5,1))
        canv.save()
        #break
