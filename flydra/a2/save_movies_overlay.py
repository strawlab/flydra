# I think this script is similar to
# flydra/analysis/flydra_analysis_plot_kalman_2d.py but better. I
# wrote that one a long time ago. - ADS 20070112

from __future__ import division
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])

import sets, os, sys, math

import pkg_resources
import numpy
import tables as PT
from optparse import OptionParser
import flydra.reconstruct as reconstruct
import sets
import motmot.ufmf.ufmf as ufmf
import flydra.a2.utils as utils

#PLOT='mpl'
PLOT='image'

if PLOT=='mpl':
    import matplotlib
    matplotlib.use('Agg')
    import pylab
    import matplotlib.cm as cm
elif PLOT=='image':
    import Image
    import aggdraw
import motmot.FlyMovieFormat.FlyMovieFormat as FMF
import flydra.analysis.result_utils as result_utils
import progressbar
import core_analysis

import pytz, datetime
pacific = pytz.timezone('US/Pacific')

def ensure_minsize_image( arr, (h,w), fill=0):
    if ((arr.shape[0] < h) or (arr.shape[1] < w)):
        arr_new = numpy.ones( (h,w), dtype=arr.dtype )*fill
        arr_new[:arr.shape[0],:arr.shape[1]] = arr
        arr=arr_new
    return arr

def doit(fmf_filename=None,
         h5_filename=None,
         kalman_filename=None,
         fps=None,
         use_kalman_smoothing=True,
         dynamic_model = None,
         start=None,
         stop=None,
         style='debug',
         blank=None,
         prefix=None,
         do_zoom=False,
         do_zoom_diff=False,
         ):
    if do_zoom_diff and do_zoom:
        raise ValueError('can use do_zoom or do_zoom_diff, but not both')

    styles = ['debug','pretty','blank']
    if style not in styles:
        raise ValueError('style ("%s") is not one of %s'%(style,str(styles)))

    if not use_kalman_smoothing:
        if (fps is not None) or (dynamic_model is not None):
            print >> sys.stderr, 'ERROR: disabling Kalman smoothing (--disable-kalman-smoothing) is incompatable with setting fps and dynamic model options (--fps and --dynamic-model)'
            sys.exit(1)

    if fmf_filename.endswith('.ufmf'):
        fmf = ufmf.FlyMovieEmulator(fmf_filename)
    else:
        fmf = FMF.FlyMovie(fmf_filename)
    fmf_timestamps = fmf.get_all_timestamps()
    h5 = PT.openFile( h5_filename, mode='r' )

    bg_fmf_filename = os.path.splitext(fmf_filename)[0] + '_bg.fmf'
    cmp_fmf_filename = os.path.splitext(fmf_filename)[0] + '_std.fmf'

    bg_OK = False
    if os.path.exists( bg_fmf_filename):
        bg_OK = True
        try:
            bg_fmf = FMF.FlyMovie(bg_fmf_filename)
            cmp_fmf = FMF.FlyMovie(cmp_fmf_filename)
        except FMF.InvalidMovieFileException, err:
            bg_OK = False

    if bg_OK:
        bg_fmf_timestamps = bg_fmf.get_all_timestamps()
        cmp_fmf_timestamps = cmp_fmf.get_all_timestamps()
        assert numpy.all((bg_fmf_timestamps[1:] - bg_fmf_timestamps[:-1])>0) # ascending
        assert numpy.all((bg_fmf_timestamps == cmp_fmf_timestamps))
        fmf2bg = bg_fmf_timestamps.searchsorted( fmf_timestamps, side='right')-1
    else:
        print 'Not loading background movie - it does not exist or it is invalid.'
        bg_fmf = None
        cmp_fmf = None
        bg_fmf_timestamps = None
        cmp_fmf_timestamps = None
        fmf2bg = None

    if fps is None:
        fps = result_utils.get_fps( h5 )

    ca = core_analysis.CachingAnalyzer()

    if blank is not None:
        # use frame number as "blank image"
        fmf.seek(blank)
        blank_image, blank_timestamp = fmf.get_next_frame()
        fmf.seek(0)
    else:
        blank_image = 255*numpy.ones( (fmf.get_height(), fmf.get_width()), dtype=numpy.uint8)

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)

        if dynamic_model is None:
            dynamic_model = extra['dynamic_model_name']
            print 'detected file loaded with dynamic model "%s"'%dynamic_model
            if dynamic_model.startswith('EKF '):
                dynamic_model = dynamic_model[4:]
            print '  for smoothing, will use dynamic model "%s"'%dynamic_model


        if is_mat_file:
            raise ValueError('cannot use .mat file for kalman_filename '
                             'because it is missing the reconstructor '
                             'and ability to get framenumbers')

        R = reconstruct.Reconstructor(data_file)
        R = R.get_scaled( R.get_scale_factor() )

        print 'loading frame numbers for kalman objects (estimates)'
        kalman_rows = []
        for obj_id in use_obj_ids:
            my_rows = ca.load_data( obj_id, data_file,
                                    use_kalman_smoothing=use_kalman_smoothing,
                                    dynamic_model_name = dynamic_model,
                                    frames_per_second=fps,
                                    )
            kalman_rows.append(my_rows)

        if len(kalman_rows):
            kalman_rows = numpy.concatenate( kalman_rows )
            kalman_3d_frame = kalman_rows['frame']

            print 'loading frame numbers for kalman objects (observations)'
            kobs_rows = []
            for obj_id in use_obj_ids:
                my_rows = ca.load_observations( obj_id, data_file,
                                                )
                kobs_rows.append(my_rows)
            kobs_rows = numpy.concatenate( kobs_rows )
            kobs_3d_frame = kobs_rows['frame']
            print 'loaded'
        else:
            print 'WARNING: kalman filename specified, but objects found'
            kalman_filename = None

    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

    n = 0
    for cam_id in cam_id2camns.keys():
        if cam_id in fmf_filename:
            n+=1
            found_cam_id = cam_id
    if n!=1:
        print >> sys.stderr, 'Could not automatically determine cam_id from fmf_filename. Exiting'
        h5.close()
        sys.exit(1)
    cam_id = found_cam_id
    my_camns = cam_id2camns[cam_id]

    if PLOT=='mpl':
        fig = pylab.figure()
    remote_timestamps = h5.root.data2d_distorted.read(field='timestamp')
    camns = h5.root.data2d_distorted.read(field='camn')
    # find rows for all camns for this cam_id
    all_camn_cond = None
    for camn in my_camns:
        cond = camn==camns
        if all_camn_cond is None:
            all_camn_cond = cond
        else:
            all_camn_cond = all_camn_cond | cond
    camn_idx = numpy.nonzero( all_camn_cond )[0]
    cam_remote_timestamps = remote_timestamps[camn_idx]
    cam_remote_timestamps_find = utils.FastFinder( cam_remote_timestamps )

    cur_bg_idx = None

    # find frame correspondence
    print 'Finding frame correspondence... ',
    sys.stdout.flush()
    frame_match_h5 = None
    first_match = None
    for fmf_fno, timestamp in enumerate( fmf_timestamps ):
        timestamp_idx = numpy.nonzero(timestamp == remote_timestamps)[0]
        #print repr(timestamp), repr(timestamp_idx)
        idxs = numpy.intersect1d( camn_idx, timestamp_idx )
        if len(idxs):
            rows = h5.root.data2d_distorted.readCoordinates( idxs )
            frame_match_h5 = rows['frame'][0]
            if start is None:
                start = frame_match_h5
            if first_match is None:
                first_match = frame_match_h5
            if stop is not None:
                break

    if frame_match_h5 is None:
        print >> sys.stderr, "ERROR: no timestamp corresponding to .fmf '%s' for %s in '%s'"%(
            fmf_filename, cam_id, h5_filename)
        h5.close()
        sys.exit(1)

    print 'done.'

    if stop is None:
        stop = frame_match_h5
        last_match = frame_match_h5
        print 'Frames in both the .fmf movie and the .h5 data file are in range %d - %d.'%(first_match, last_match)
    else:
        print 'Frames in both the .fmf movie and the .h5 data file start at %d.'%(first_match,)

    if 1:
        #if first_match is not None and last_match is not None:
        h5frames = h5.root.data2d_distorted.read( field='frame' )[:]
        print 'The .h5 file has frames %d - %d.'%( h5frames.min(), h5frames.max() )
        del h5frames

    if start > stop:
        print >> sys.stderr, "ERROR: start (frame %d) is after stop (frame %d)!"%(start,stop)
        h5.close()
        sys.exit(1)

    fmf_frame2h5_frame = frame_match_h5 - fmf_fno

    if PLOT=='image':
        # colors from: http://jfly.iam.u-tokyo.ac.jp/color/index.html#pallet

        cb_orange = (230, 159, 0)
        cb_blue = (0, 114, 178)
        cb_vermillion = (213, 94, 0)

        font2d = aggdraw.Font(cb_blue,'/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',size=20)
        pen2d = aggdraw.Pen(cb_blue, width=2 )

        pen3d = aggdraw.Pen(cb_orange, width=2 )
        font3d = aggdraw.Font(cb_orange,'/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf')

        pen_zoomed = pen3d
        font_zoomed = aggdraw.Font(cb_orange,'/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', size=20)

        pen_obs = aggdraw.Pen(cb_vermillion, width=2 )
        font_obs = aggdraw.Font(cb_vermillion,'/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf')

    print 'loading frame information...'
    # step through .fmf file to get map of h5frame <-> fmfframe
    mymap = {}
    all_frame = h5.root.data2d_distorted.read(field='frame')
    cam_all_frame = all_frame[ camn_idx ]

    widgets=['stage 1 of 2: ',cam_id, ' ', progressbar.Percentage(), ' ',
             progressbar.Bar(), ' ', progressbar.ETA()]

    pbar=progressbar.ProgressBar(widgets=widgets,maxval=len(fmf_timestamps)).start()
    for fmf_fno, fmf_timestamp in enumerate( fmf_timestamps ):
        pbar.update(fmf_fno)
        # idxs = numpy.nonzero(cam_remote_timestamps==fmf_timestamp)[0]
        idxs = cam_remote_timestamps_find.get_idxs_of_equal(fmf_timestamp)
        if len(idxs):
            this_frame = cam_all_frame[idxs]
            real_h5_frame = int(this_frame[0])
            # we only should have one frame here
            assert numpy.all( real_h5_frame== this_frame )
            mymap[real_h5_frame]= fmf_fno
    pbar.finish()
    print 'done loading frame information.'

    print 'start, stop',start, stop
    widgets[0]='stage 2 of 2: '
    pbar=progressbar.ProgressBar(widgets=widgets,maxval=(stop-start+1)).start()
    for h5_frame in range(start,stop+1):
        pbar.update(h5_frame-start)
        mainbrain_timestamp = numpy.nan
        idxs = []
        try:
            fmf_fno = mymap[h5_frame]
            fmf_timestamp = fmf_timestamps[fmf_fno]

            # get frame
            fmf.seek(fmf_fno)
            frame, fmf_timestamp2 = fmf.get_next_frame()
            assert fmf_timestamp==fmf_timestamp2, "timestamps of .fmf frames don't match"

            # get bg frame
            if fmf2bg is not None:
                bg_idx = fmf2bg[fmf_fno]
                if cur_bg_idx != bg_idx:
                    bg_frame, bg_timestamp = bg_fmf.get_frame(bg_idx)
                    cmp_frame, trash = cmp_fmf.get_frame(bg_idx)
                    cur_bg_idx = bg_idx

            timestamp_idx = numpy.nonzero(fmf_timestamp == remote_timestamps)[0]
            idxs = numpy.intersect1d( camn_idx, timestamp_idx )
            rows = None
            if len(idxs):
                rows = h5.root.data2d_distorted.readCoordinates( idxs )
                mainbrain_timestamp = rows['cam_received_timestamp'][0]

            del fmf_fno
            del fmf_timestamp2
        except KeyError,err:
            frame = blank_image

        if PLOT=='image':
            im = None

        if 1:
            # get 3D estimates data
            kalman_vert_images = []
            if kalman_filename is not None:
                data_3d_idxs = numpy.nonzero(h5_frame == kalman_3d_frame)[0]
                these_3d_rows = kalman_rows[data_3d_idxs]
                for this_3d_row in these_3d_rows:
                    vert = numpy.array([this_3d_row['x'],this_3d_row['y'],this_3d_row['z']])
                    vert_image = R.find2d(cam_id,vert,distorted=True)
                    P = numpy.array([this_3d_row['P00'],this_3d_row['P11'],this_3d_row['P22']])
                    Pmean = numpy.sqrt(numpy.sum(P**2))
                    Pmean_meters = numpy.sqrt(Pmean)
                    kalman_vert_images.append( (vert_image, vert, this_3d_row['obj_id'], Pmean_meters) )

            # get 3D observation data
            kobs_vert_images = []
            if kalman_filename is not None:
                data_3d_idxs = numpy.nonzero(h5_frame == kobs_3d_frame)[0]
                these_3d_rows = kobs_rows[data_3d_idxs]
                for this_3d_row in these_3d_rows:
                    vert = numpy.array([this_3d_row['x'],this_3d_row['y'],this_3d_row['z']])
                    vert_image = R.find2d(cam_id,vert,distorted=True)
                    obs_2d_idx = this_3d_row['obs_2d_idx']
                    kobs_2d_data = data_file.root.kalman_observations_2d_idxs[int(obs_2d_idx)]

                    # parse VLArray
                    this_camns = kobs_2d_data[0::2]
                    this_camn_idxs = kobs_2d_data[1::2]
                    this_cam_ids = [camn2cam_id[this_camn] for this_camn in this_camns]
                    obs_info = (this_cam_ids, this_camn_idxs)

                    kobs_vert_images.append( (vert_image, vert, this_3d_row['obj_id'], obs_info) )

            if do_zoom_diff or do_zoom:
                fg = frame.astype(numpy.float32)
                zoom_fgs = []
                obj_ids = []
                this2ds = []
                if do_zoom_diff:
                    zoom_luminance_scale = 7.0
                    zoom_luminance_offset = 127
                    zoom_scaled_black = -zoom_luminance_offset/zoom_luminance_scale

                    # Zoomed difference image for this frame
                    bg = bg_frame.astype(numpy.float32)
                    cmp = cmp_frame.astype(numpy.float32)
                    diff_im = fg-bg

                    zoom_diffs = []
                    zoom_absdiffs = []
                    zoom_bgs = []
                    zoom_cmps = []

                    maxabsdiff = []
                    maxcmp = []
                plotted_pt_nos = sets.Set()
                radius=10
                h,w = fg.shape
                for (xy,XYZ,obj_id,Pmean_meters) in kalman_vert_images:
                    x,y= xy
                    if ((0 <= x <= w) and
                        (0 <= y <= h)):
                        minx = max(0,x-radius)
                        maxx = min(w,minx+(2*radius))
                        miny = max(0,y-radius)
                        maxy = min(h,miny+(2*radius))

                        zoom_fg = fg[miny:maxy, minx:maxx]
                        zoom_fg =  ensure_minsize_image( zoom_fg,  (2*radius, 2*radius ))
                        zoom_fgs.append( zoom_fg )

                        if do_zoom_diff:
                            zoom_diff = diff_im[miny:maxy, minx:maxx]
                            zoom_diff = ensure_minsize_image( zoom_diff, (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_diffs.append( zoom_diff )
                            zoom_absdiffs.append( abs(zoom_diff) )

                            zoom_bg = bg[miny:maxy, minx:maxx]
                            zoom_bg =  ensure_minsize_image( zoom_bg,  (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_bgs.append( zoom_bg )

                            zoom_cmp = cmp[miny:maxy, minx:maxx]
                            zoom_cmp =  ensure_minsize_image( zoom_cmp,  (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_cmps.append( zoom_cmp )

                            maxabsdiff.append( abs(zoom_diff ).max() )
                            maxcmp.append( zoom_cmp.max() )

                        obj_ids.append( obj_id )

                        this2d = []
                        for pt_no, (x2d,y2d) in enumerate(zip(rows['x'],rows['y'])):
                            if ((minx <= x2d <= maxx) and
                                (miny <= y2d <= maxy)):
                                this2d.append( (x2d-minx,y2d-miny,pt_no) )
                                plotted_pt_nos.add( int(pt_no) )
                        this2ds.append( this2d )
                all_pt_nos = sets.Set(range( len( rows) ))
                missing_pt_nos = list(all_pt_nos-plotted_pt_nos)

                for pt_no in missing_pt_nos:
                    this_row = rows[pt_no]
                    (x2d,y2d) = this_row['x'], this_row['y']
                    x=x2d
                    y=y2d

                    if ((0 <= x <= w) and
                        (0 <= y <= h)):
                        minx = max(0,x-radius)
                        maxx = min(w,minx+(2*radius))
                        miny = max(0,y-radius)
                        maxy = min(h,miny+(2*radius))

                        zoom_fg = fg[miny:maxy, minx:maxx]
                        zoom_fg =  ensure_minsize_image( zoom_fg,  (2*radius, 2*radius ))
                        zoom_fgs.append( zoom_fg )

                        if do_zoom_diff:
                            zoom_diff = diff_im[miny:maxy, minx:maxx]
                            zoom_diff = ensure_minsize_image( zoom_diff, (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_diffs.append( zoom_diff )
                            zoom_absdiffs.append( abs(zoom_diff) )

                            zoom_bg = bg[miny:maxy, minx:maxx]
                            zoom_bg =  ensure_minsize_image( zoom_bg,  (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_bgs.append( zoom_bg )

                            zoom_cmp = cmp[miny:maxy, minx:maxx]
                            zoom_cmp =  ensure_minsize_image( zoom_cmp,  (2*radius, 2*radius ),fill=zoom_scaled_black)
                            zoom_cmps.append( zoom_cmp )

                            maxabsdiff.append( abs(zoom_diff ).max() )
                            maxcmp.append( zoom_cmp.max() )

                        obj_ids.append( None )

                        this2d = []
                        if 1:
                            if ((minx <= x2d <= maxx) and
                                (miny <= y2d <= maxy)):
                                this2d.append( (x2d-minx,y2d-miny,pt_no) )
                                plotted_pt_nos.add( int(pt_no) )
                        this2ds.append( this2d )

                if len(zoom_fgs):
                    top_offset = 5
                    left_offset = 30
                    fgrow = numpy.hstack( zoom_fgs )
                    blackrow = numpy.zeros( (top_offset, fgrow.shape[1]) )
                    if do_zoom_diff:
                        scale = zoom_luminance_scale
                        offset = zoom_luminance_offset

                        diffrow = numpy.hstack( zoom_diffs ) * scale + offset
                        cmprow = numpy.hstack( zoom_cmps ) * scale + offset

                        bgrow = numpy.hstack( zoom_bgs )
                        absdiffrow = numpy.hstack( zoom_absdiffs ) * scale+offset

                        row_ims=[blackrow,
                                 diffrow,
                                 cmprow,
                                 absdiffrow,
                                 blackrow,
                                 fgrow,
                                 bgrow,
                                 ]
                        labels = [None,
                                  'diff (s.)',
                                  'cmp (s.)',
                                  'absdiff (s.)',
                                  None,
                                  'raw',
                                  'bg',
                                  ]

                    else:
                        row_ims=[blackrow,
                                 fgrow,
                                 ]
                        labels = [None,
                                  'raw',
                                  ]

                    rightpart = numpy.vstack(row_ims)
                    leftpart = numpy.zeros( (rightpart.shape[0], left_offset) )
                    newframe = numpy.hstack( [leftpart, rightpart] )

                    newframe = numpy.clip( newframe, 0, 255)
                    im = newframe.astype( numpy.uint8 ) # scale and offset
                    im=Image.fromstring('L',
                                        (im.shape[1],im.shape[0]),
                                        im.tostring())
                    w,h = im.size
                    rescale_factor = 5
                    im = im.resize( (rescale_factor*w, rescale_factor*h) )
                    im = im.convert('RGB')
                    draw = aggdraw.Draw(im)

                    cumy = 0
                    absdiffy = None
                    cmpy = None
                    for j in range(len(labels)):
                        label = labels[j]
                        row_im = row_ims[j]
                        draw.text( (rescale_factor*(0),rescale_factor*cumy),
                                   label, font_zoomed)
                        if (do_zoom_diff and label == 'absdiff (s.)'):
                            absdiffy = cumy
                        elif (do_zoom and label == 'raw'):
                            absdiffy = cumy
                        if label == 'cmp (s.)':
                            cmpy = cumy
                        cumy += row_im.shape[0]

                    for i, obj_id in enumerate(obj_ids):
                        if obj_id is not None:
                            draw.text( (rescale_factor*(i*2*radius+left_offset),0),
                                       'obj %d'%(obj_id,), font_zoomed)

                    if do_zoom_diff:
                        for i, (this_maxabsdiff, this_maxcmp) in enumerate(
                            zip(maxabsdiff, maxcmp) ):

                            draw.text( (rescale_factor*(i*2*radius+left_offset)+3,
                                        rescale_factor*(absdiffy)),
                                       'max %.0f'%(this_maxabsdiff,), font_zoomed)

                            draw.text( (rescale_factor*(i*2*radius+left_offset)+3,
                                        rescale_factor*(cmpy)),
                                       'max %.0f'%(this_maxcmp,), font_zoomed)

                    radius_pt = 3
                    for i,this2d in enumerate(this2ds):
                        for (x2d, y2d, pt_no) in this2d:
                            xloc = rescale_factor*( x2d + i*2*radius + left_offset )
                            xloc1 = xloc - radius_pt
                            xloc2 = xloc + radius_pt
                            draw.ellipse( [xloc1,rescale_factor*(absdiffy+y2d)-radius_pt,
                                           xloc2,rescale_factor*(absdiffy+y2d)+radius_pt],
                                          pen_zoomed )
                            draw.text( (xloc,rescale_factor*(absdiffy+y2d)),
                                       'pt %d'%(pt_no,), font_zoomed )
                    draw.flush()
                    fname = 'zoomed/zoom_diff_%(cam_id)s_%(h5_frame)07d.png'%locals()
                    dirname = os.path.abspath(os.path.split(fname)[0])
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    im.save( fname )

            if PLOT=='mpl':
                pylab.imshow( frame,
                              origin='lower',
                              cmap=cm.pink,
                              )

                ylim = pylab.gca().get_ylim()
                pylab.gca().set_ylim((ylim[1],ylim[0]))

                if len(idxs):
                    x = rows['x']
                    y = rows['y']
                    if not numpy.isnan(x[0]):
                        pylab.plot(x,y,'.')
                if len(kalman_vert_images):
                    raise NotImplementedError('')

            elif PLOT=='image':
                assert fmf.format=='MONO8'
                im=Image.fromstring('L',
                                    (frame.shape[1],frame.shape[0]),
                                    frame.tostring())
                im = im.convert('RGB')
                draw = aggdraw.Draw(im)

                if style=='debug':
                    try:
                        strtime = datetime.datetime.fromtimestamp(mainbrain_timestamp,pacific)
                    except:
                        strtime = '<no 2d data timestamp>'
                    draw.text( (0,0), 'frame %d, %s timestamp %s - %s'%(
                        h5_frame, cam_id, repr(fmf_timestamp), strtime), font2d )

                if len(idxs):
                    for pt_no,(x,y,area,slope,eccentricity) in enumerate(zip(rows['x'],
                                                                             rows['y'],
                                                                             rows['area'],rows['slope'],
                                                                             rows['eccentricity'])):
                        if style=='debug':
                            radius = numpy.sqrt(area/(2*numpy.pi))
                            draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                          pen2d )

                            pos = numpy.array( [x,y] )
                            if 0:
                                direction = numpy.array( [slope,1] )
                                direction = direction/numpy.sqrt(numpy.sum(direction**2)) # normalize
                                vec = direction*eccentricity
                                p1 = pos+vec
                                p2 = pos-vec
                                draw.line(    [p1[0],p1[1], p2[0],p2[1]],
                                              pen2d )
                            tmp_str = 'pt %d (area %f)'%(pt_no,area)
                            tmpw,tmph = draw.textsize(tmp_str, font2d )
                            draw.text( (x+5,y-tmph-1), tmp_str, font2d )
                        elif style=='pretty':
                            radius = 30
                            draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                          pen2d )

                for (xy,XYZ,obj_id,Pmean_meters) in kalman_vert_images:
                    if style in ['debug','pretty']:
                        radius=10
                        x,y= xy
                        X,Y,Z=XYZ
                        draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                      pen3d )
                    if style=='debug':
                        draw.text( (x+5,y), 'obj %d (%.3f, %.3f, %.3f +- ~%f)'%(obj_id,X,Y,Z,Pmean_meters), font3d )

                if style=='debug':
                    for (xy,XYZ,obj_id,obs_info) in kobs_vert_images:
                        radius=3
                        x,y= xy
                        X,Y,Z=XYZ
                        draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                      pen_obs )
                        draw.text( (x+5,y), 'obj %d (%.3f, %.3f, %.3f)'%(obj_id,X,Y,Z), font_obs )
                        (this_cam_ids, this_camn_idxs) = obs_info
                        for i,(obs_cam_id,pt_no) in enumerate( zip(*obs_info) ):
                            draw.text( (x+15,y+(i+1)*10),
                                       '%s pt %d'%(obs_cam_id,pt_no), font_obs )


                draw.flush()

        if 1:
            fname = 'full/smo_%(cam_id)s_%(h5_frame)07d.png'%locals()
            if prefix is not None:
                fname = prefix + '_' + fname
            dirname = os.path.abspath(os.path.split(fname)[0])
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            #print 'saving',fname
            if PLOT=='mpl':
                fig.savefig( fname )
            elif PLOT=='image':
                if im is not None:
                    im.save( fname )

        if PLOT=='mpl':
            fig.clear()
    pbar.finish()

    h5.close()

def main():
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--fmf", dest="fmf_filename", type='string',
                      help=".fmf filename (REQUIRED)")

    parser.add_option("--h5", dest="h5_filename", type='string',
                      help=".h5 file with data2d_distorted (REQUIRED)")

    parser.add_option("--kalman", dest="kalman_filename", type='string',
                      help=".h5 file with kalman data and 3D reconstructor")

    parser.add_option("--prefix", dest="prefix", type='string',
                      help="prefix for output image filenames")

    parser.add_option("--start", dest="start", type='int',
                      help="start frame (.h5 frame number reference)")

    parser.add_option("--stop", dest="stop", type='int',
                      help="stop frame (.h5 frame number reference)")

    parser.add_option("--blank", dest="blank", type='int',
                      help="frame number of FMF file (fmf-reference) of blank image to use when no image")

    parser.add_option("--disable-kalman-smoothing", action='store_false',dest='use_kalman_smoothing',
                      default=True,
                      help="show original, causal Kalman filtered data (rather than Kalman smoothed observations)")

    parser.add_option("--fps", dest='fps', type='float',
                      help="frames per second (used for Kalman filtering/smoothing)")

    parser.add_option("--dynamic-model",
                      type="string",
                      dest="dynamic_model",
                      default=None,
                      )

    parser.add_option("--style", dest="style", type='string',
                      default='debug',)

    parser.add_option("--zoom", action='store_true',
                      default=False,
                      help="save zoomed image (not well tested)")

    parser.add_option("--zoom-diff", action='store_true',
                      default=False,
                      help="save zoomed difference image (not well tested)")

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit(fmf_filename=options.fmf_filename,
         h5_filename=options.h5_filename,
         kalman_filename=options.kalman_filename,
         fps = options.fps,
         dynamic_model = options.dynamic_model,
         use_kalman_smoothing=options.use_kalman_smoothing,
         start=options.start,
         stop=options.stop,
         style=options.style,
         blank=options.blank,
         prefix=options.prefix,
         do_zoom=options.zoom,
         do_zoom_diff=options.zoom_diff,
         )

if __name__=='__main__':
    main()
