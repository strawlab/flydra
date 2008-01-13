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

#PLOT='mpl'
PLOT='image'
SKIP_NODATA = True

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

def doit(fmf_filename=None,
         h5_filename=None,
         kalman_filename=None,
         start=None,
         stop=None,
         ):
    fmf = FMF.FlyMovie(fmf_filename)
    h5 = PT.openFile( h5_filename, mode='r' )
    ca = core_analysis.CachingAnalyzer()

    if kalman_filename is not None:
        obj_ids, use_obj_ids, is_mat_file, data_file, extra = ca.initial_file_load(kalman_filename)
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
                                    use_kalman_smoothing=False,
                                    #use_kalman_smoothing=use_kalman_smoothing,
                                    #kalman_dynamic_model = dynamic_model,
                                    #frames_per_second=fps,
                                    )
            kalman_rows.append(my_rows)
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

    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(h5)

    n = 0
    for cam_id in cam_id2camns.keys():
        if cam_id in fmf_filename:
            n+=1
            found_cam_id = cam_id
    if n!=1:
        print >> sys.stderr, 'Could not automatically determine cam_id from fmf_filename. Exiting'
        sys.exit(1)
    cam_id = found_cam_id
    camns = cam_id2camns[cam_id]
    assert len(camns)==1
    camn = camns[0]

    if PLOT=='mpl':
        fig = pylab.figure()
    timestamps = h5.root.data2d_distorted.read(field='timestamp')
    camns = h5.root.data2d_distorted.read(field='camn')
    camn_idx = numpy.nonzero(camn==camns)[0]

    fmf_timestamps = fmf.get_all_timestamps()
    # find frame correspondence
    frame_match_h5 = None
    for fmf_fno, timestamp in enumerate( fmf_timestamps ):
        timestamp_idx = numpy.nonzero(timestamp == timestamps)[0]
        #print repr(timestamp), repr(timestamp_idx)
        idxs = numpy.intersect1d( camn_idx, timestamp_idx )
        if len(idxs):
            rows = h5.root.data2d_distorted.readCoordinates( idxs )
            frame_match_h5 = rows['frame'][0]
            break
    #print
    if frame_match_h5 is None:
        print >> sys.stderr, "ERROR: no timestamp corresponding to .fmf '%s' for %s in '%s'"%(
            fmf_filename, cam_id, h5_filename)
        sys.exit(1)

    fmf_frame2h5_frame = frame_match_h5 - fmf_fno

    widgets=[cam_id, " ", progressbar.Percentage(), ' ',
             progressbar.Bar(), ' ', progressbar.ETA()]
    pbar=progressbar.ProgressBar(widgets=widgets,maxval=len(fmf_timestamps)).start()

    if PLOT=='image':
        font = aggdraw.Font('Lime','/usr/share/fonts/truetype/freefont/FreeMono.ttf')
        pen = aggdraw.Pen('Lime', width=2 )

        pen3d = aggdraw.Pen('Red', width=2 )
        font3d = aggdraw.Font('Red','/usr/share/fonts/truetype/freefont/FreeMono.ttf')

        pen_obs = aggdraw.Pen('Blue', width=2 )
        font_obs = aggdraw.Font('Blue','/usr/share/fonts/truetype/freefont/FreeMono.ttf')

    # step through .fmf file
    for fmf_fno, timestamp in enumerate( fmf_timestamps ):
        pbar.update(fmf_fno)
        h5_frame = fmf_fno + fmf_frame2h5_frame
        timestamp_idx = numpy.nonzero(timestamp == timestamps)[0]
        #print 'ts', fmf_fno,len(timestamp_idx)
        if SKIP_NODATA and not len(timestamp_idx):
            continue # no point in plotting images without data in .h5 file
        idxs = numpy.intersect1d( camn_idx, timestamp_idx )

        #print fmf_fno,len(idxs)

        if PLOT=='image':
            im = None

        rows = None
        mainbrain_timestamp = numpy.nan
        if len(idxs):
            rows = h5.root.data2d_distorted.readCoordinates( idxs )
            mainbrain_timestamp = rows['cam_received_timestamp'][0]
            if not numpy.all( h5_frame==rows['frame'] ):
                real_h5_frame = rows['frame'][0]
                try:
                    # We may have skipped a frame saving movie, so
                    # h5_frame can be less than actual.
                    assert numpy.all( h5_frame<= rows['frame'] )

                    assert numpy.all( real_h5_frame== rows['frame'] )
                except:
                    print "h5_frame",h5_frame
                    print "rows['frame']",rows['frame']
                    raise
                n_skip = real_h5_frame-h5_frame
                h5_frame = real_h5_frame
                fmf_frame2h5_frame += n_skip
                #print 'skipped %d frame(s) in .fmf'%(n_skip,)

        if start is not None:
            if h5_frame < start:
                continue

        if stop is not None:
            if h5_frame > stop:
                continue

        if 1:
            # get frame
            fmf.seek(fmf_fno)
            frame, timestamp2 = fmf.get_next_frame()
            assert timestamp==timestamp2

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

                strtime = datetime.datetime.fromtimestamp(mainbrain_timestamp,pacific)
                draw.text( (0,0), 'frame %d, %s (%d) timestamp %s - %s'%(
                    h5_frame, cam_id, camn, repr(timestamp), strtime), font )

                if len(idxs):
                    for pt_no,(x,y,area,slope,eccentricity) in enumerate(zip(rows['x'],
                                                                             rows['y'],
                                                                             rows['area'],rows['slope'],
                                                                             rows['eccentricity'])):
                        radius = numpy.sqrt(area/(2*numpy.pi))

                        pos = numpy.array( [x,y] )
                        direction = numpy.array( [slope,1] )
                        direction = direction/numpy.sqrt(numpy.sum(direction**2)) # normalize
                        vec = direction*eccentricity
                        p1 = pos+vec
                        p2 = pos-vec
                        draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                      pen )
                        draw.line(    [p1[0],p1[1], p2[0],p2[1]],
                                      pen )
                        draw.text( (x,y), 'pt %d (area %f)'%(pt_no,area), font )

                for (xy,XYZ,obj_id,Pmean_meters) in kalman_vert_images:
                    radius=3
                    x,y= xy
                    X,Y,Z=XYZ
                    draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                  pen3d )
                    draw.text( (x,y), 'obj %d (%.3f, %.3f, %.3f +- ~%f)'%(obj_id,X,Y,Z,Pmean_meters), font3d )

                for (xy,XYZ,obj_id,obs_info) in kobs_vert_images:
                    radius=3
                    x,y= xy
                    X,Y,Z=XYZ
                    draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                  pen_obs )
                    draw.text( (x,y), 'obj %d (%.3f, %.3f, %.3f)'%(obj_id,X,Y,Z), font_obs )
                    (this_cam_ids, this_camn_idxs) = obs_info
                    for i,(obs_cam_id,pt_no) in enumerate( zip(*obs_info) ):
                        draw.text( (x+10,y+(i+1)*10),
                                   '%s pt %d'%(obs_cam_id,pt_no), font_obs )


                draw.flush()

        if 1:
            fname = 'smo_%s_%07d.png'%(cam_id,h5_frame)
            #print 'saving',fname
            if PLOT=='mpl':
                fig.savefig( fname )
            elif PLOT=='image':
                if im is not None:
                    im.save( 'PIL_'+fname )

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

    parser.add_option("--start", dest="start", type='int',
                      help="start frame (.h5 frame number reference)")

    parser.add_option("--stop", dest="stop", type='int',
                      help="stop frame (.h5 frame number reference)")

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit(fmf_filename=options.fmf_filename,
         h5_filename=options.h5_filename,
         kalman_filename=options.kalman_filename,
         start=options.start,
         stop=options.stop,
         )

if __name__=='__main__':
    main()
