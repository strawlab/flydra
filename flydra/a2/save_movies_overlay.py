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
import flydra.analysis.result_utils
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

def doit(fmf_filename=None,
         h5_filename=None,
         kalman_filename=None,
         start=None,
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

        print 'loading frame numbers for kalman objects'
        all_rows = []
        for obj_id in use_obj_ids:
            my_rows = ca.load_data( obj_id, data_file,
                                    use_kalman_smoothing=False,
                                    #use_kalman_smoothing=use_kalman_smoothing,
                                    #kalman_dynamic_model = dynamic_model,
                                    #frames_per_second=fps,
                                    )
            all_rows.append(my_rows)
        all_rows = numpy.concatenate( all_rows )
        data_3d_frame = all_rows['frame']

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
    for fmf_fno, timestamp in enumerate( fmf_timestamps ):
        timestamp_idx = numpy.nonzero(timestamp == timestamps)[0]
        #print repr(timestamp), repr(timestamp_idx)
        idxs = numpy.intersect1d( camn_idx, timestamp_idx )
        if len(idxs):
            rows = h5.root.data2d_distorted.readCoordinates( idxs )
            frame_match_h5 = rows['frame'][0]
            break
    #print

    fmf_frame2h5_frame = frame_match_h5 - fmf_fno

    widgets=["calculating", " ", progressbar.Percentage(), ' ',
             progressbar.Bar(), ' ', progressbar.ETA()]
    pbar=progressbar.ProgressBar(widgets=widgets,maxval=len(fmf_timestamps)).start()

    if PLOT=='image':
        font = aggdraw.Font('Lime','/usr/share/fonts/truetype/freefont/FreeMono.ttf')
        pen = aggdraw.Pen('Lime', width=2 )

        pen3d = aggdraw.Pen('Red', width=2 )
        font3d = aggdraw.Font('Red','/usr/share/fonts/truetype/freefont/FreeMono.ttf')

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
        if len(idxs):
            rows = h5.root.data2d_distorted.readCoordinates( idxs )

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

        if 1:
            # get frame
            fmf.seek(fmf_fno)
            frame, timestamp2 = fmf.get_next_frame()
            assert timestamp==timestamp2

            # get 3D data
            vert_images = []
            if kalman_filename is not None:
                data_3d_idxs = numpy.nonzero(h5_frame == data_3d_frame)[0]
                these_3d_rows = all_rows[data_3d_idxs]
                for this_3d_row in these_3d_rows:
                    vert = numpy.array([this_3d_row['x'],this_3d_row['y'],this_3d_row['z']])
                    vert_image = R.find2d(cam_id,vert,distorted=True)
                    vert_images.append( (vert_image, this_3d_row['obj_id']) )

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
                if len(vert_images):
                    raise NotImplementedError('')

            elif PLOT=='image':
                assert fmf.format=='MONO8'
                im=Image.fromstring('L',
                                    (frame.shape[1],frame.shape[0]),
                                    frame.tostring())
                im = im.convert('RGB')
                draw = aggdraw.Draw(im)

                draw.text( (0,0), 'frame %d, timestamp %s, cam %s'%(
                    h5_frame, repr(timestamp), cam_id), font )

                if len(idxs):
                    radius=3
                    for x,y in zip(rows['x'],rows['y']):
                        draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                      pen )
                for (xy,obj_id) in vert_images:
                    radius=3
                    x,y= xy
                    draw.ellipse( [x-radius,y-radius,x+radius,y+radius],
                                  pen3d )
                    draw.text( (x,y), 'obj %d'%obj_id, font3d )

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

    (options, args) = parser.parse_args()

    if len(args):
        parser.print_help()
        return

    doit(fmf_filename=options.fmf_filename,
         h5_filename=options.h5_filename,
         kalman_filename=options.kalman_filename,
         start=options.start,
         )

if __name__=='__main__':
    main()
