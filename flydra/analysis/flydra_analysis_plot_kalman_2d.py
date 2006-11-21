from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sets
import sys
from optparse import OptionParser
import pylab
import flydra.reconstruct
import flydra.analysis.result_utils as result_utils

def auto_subplot(fig,n,n_rows=2,n_cols=3):
    # 2 rows and n_cols
    
    rrow = n // n_cols # reverse row
    row = n_rows-rrow-1 # row number
    col = n % n_cols

    x_space = (0.02/n_cols)
    #y_space = 0.0125
    y_space = 0.03
    y_size = (1.0/n_rows)-(2*y_space)
    
    left = col*(1.0/n_cols) + x_space
    bottom = row*y_size + y_space
    w = (1.0/n_cols) - x_space
    h = y_size - 2*y_space
    return fig.add_axes([left,bottom,w,h])

def show_it(fig,
            filename,
            kalman_filename = None,
            frame_start = None,
            frame_stop = None,
            animate = False,
            ):

    results = result_utils.get_results(filename,mode='r')
    reconstructor = flydra.reconstruct.Reconstructor(results)
    
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
    
    data2d = results.root.data2d_distorted # make sure we have 2d data table
    
    if frame_start is not None:
        print 'selecting frames after start'
        after_start = data2d.getWhereList( data2d.cols.frame>=frame_start,
                                          flavor='numpy' )
    else:
        after_start = None
        
    if frame_stop is not None:
        print 'selecting frames before stop'
        before_stop = data2d.getWhereList( data2d.cols.frame<=frame_stop,
                                           flavor='numpy' )
    else:
        before_stop = None

    print 'finding all frames'
    if after_start is not None and before_stop is not None:
        use_idxs = numpy.intersect1d(after_start,before_stop)
    elif after_start is not None:
        use_idxs = after_start
    elif before_stop is not None:
        use_idxs = before_stop
    else:
        use_idxs = numpy.arange(data2d.nrows)

        
    # OK, we have data coords, plot

    print 'reading cameras'
    frames = data2d.readCoordinates( use_idxs, field='frame', flavor='numpy')
    camns = data2d.readCoordinates( use_idxs, field='camn', flavor='numpy')
    unique_camns = numpy.unique1d(camns)
    unique_cam_ids = list(sets.Set([camn2cam_id[camn] for camn in unique_camns]))
    unique_cam_ids.sort()
    print '%d cameras with data'%(len(unique_cam_ids),)
    if animate:
        raise NotImplementedError('xxx')

    subplot_by_cam_id = {}
    for i,cam_id in enumerate(unique_cam_ids):
        ax = auto_subplot(fig,i)
        ax.text(0.5,0.95,cam_id,
                horizontalalignment='center',
                verticalalignment='top',
                transform = ax.transAxes,
                )
##        ax.set_xticks([])
##        ax.set_yticks([])
        subplot_by_cam_id[cam_id] = ax


    for camn in unique_camns:
        cam_id = camn2cam_id[camn]
        ax = subplot_by_cam_id[cam_id]
        this_camn_idxs = use_idxs[camns == camn]
        
        xs = data2d.readCoordinates( this_camn_idxs, field='x', flavor='numpy')
        ys = data2d.readCoordinates( this_camn_idxs, field='y', flavor='numpy')

        ax.plot(xs,ys,'.')
        
        if reconstructor is not None:
            res = reconstructor.get_resolution(cam_id)
            ax.set_xlim([0,res[0]])
            ax.set_ylim([0,res[1]])

    # Do same as above for Kalman-filtered data

    if kalman_filename is None:
        return
    
    kresults = PT.openFile(kalman_filename,mode='r')
    kobs = kresults.root.kalman_observations
    if frame_start is not None:
        k_after_start = kobs.getWhereList(
            kobs.cols.frame>=frame_start,
            flavor='numpy' )
    else:
        k_after_start = None
    if frame_stop is not None:
        k_before_stop = kobs.getWhereList(
            kobs.cols.frame<=frame_stop,
            flavor='numpy' )
    else:
        k_before_stop = None

    if k_after_start is not None and k_before_stop is not None:
        k_use_idxs = numpy.intersect1d(k_after_start,k_before_stop)
    elif k_after_start is not None:
        k_use_idxs = k_after_start
    elif k_before_stop is not None:
        k_use_idxs = k_before_stop
    else:
        k_use_idxs = numpy.arange(data2d.nrows)


    obj_ids = kobs.readCoordinates( k_use_idxs,
                                    field='obj_id',
                                    flavor='numpy')
    obs_2d_idxs = kobs.readCoordinates( k_use_idxs,
                                        field='obs_2d_idx',
                                        flavor='numpy')
    kframes = kobs.readCoordinates( k_use_idxs,
                                   field='frame',
                                   flavor='numpy')
    
    kobs_2d = kresults.root.kalman_observations_2d_idxs
    xys_by_obj_id = {}
    for obj_id,kframe,obs_2d_idx in zip(obj_ids,kframes,obs_2d_idxs):
        if PT.__version__ <= '1.3.3':
            obs_2d_idx_find = int(obs_2d_idx)
        else:
            obs_2d_idx_find = obs_2d_idx
        obj_id_save = int(obj_id) # convert from possible numpy scalar
        xys_by_cam_id = xys_by_obj_id.setdefault( obj_id_save, {})
        
        kobs_2d_data = kobs_2d.read( start=obs_2d_idx_find,
                                     stop=obs_2d_idx_find+1 )
        assert len(kobs_2d_data)==1
        kobs_2d_data = kobs_2d_data[0]
        this_camns = kobs_2d_data[0::2]
        this_camn_idxs = kobs_2d_data[1::2]

        this_use_idxs = use_idxs[frames==kframe]

        d2d = data2d.readCoordinates( this_use_idxs, flavor='numpy')
        for this_camn,this_camn_idx in zip(this_camns,this_camn_idxs):
            this_camn_d2d = d2d[d2d.camn == this_camn]
            found = False
            for this_row in this_camn_d2d: # XXX could be sped up
                if this_row.frame_pt_idx == this_camn_idx:
                    found = True
                    break
            if not found:
                raise RuntimeError('point not found in data!?')
            #this_row = this_camn_d2d[this_camn_idx]
            this_cam_id = camn2cam_id[this_camn]
            xys = xys_by_cam_id.setdefault( this_cam_id, ([],[]) )
            xys[0].append( this_row.x )
            xys[1].append( this_row.y )
            if obj_id_save == 286:
                print this_cam_id
                print this_camn_idx
                print this_camn_d2d
                print this_row.x, this_row.y
                print
            
    for obj_id in xys_by_obj_id:
        xys_by_cam_id = xys_by_obj_id[obj_id]
        for cam_id, (xs,ys) in xys_by_cam_id.iteritems():
            xs, ys = xys_by_cam_id[cam_id]
            ax = subplot_by_cam_id[cam_id]
            ax.plot(xs,ys,'o-')
            ax.text(xs[0],ys[0],'%d'%(obj_id,))
        
def main():
    usage = '%prog FILE [options]'
    
    parser = OptionParser(usage)
    
    parser.add_option("-f", "--file", dest="filename", type='string',
                      help="hdf5 file with data to display FILE",
                      metavar="FILE")

    parser.add_option("--kalman-file", dest="kalman_filename", type='string',
                      help="hdf5 file with kalman data to display KALMANFILE",
                      metavar="KALMANFILE")

    parser.add_option("--start", type="int",
                      help="first frame to plot",
                      metavar="START")
        
    parser.add_option("--stop", type="int",
                      help="last frame to plot",
                      metavar="STOP")
    
    parser.add_option("--animate", action='store_true',dest='animate',
                      help="animate")

    (options, args) = parser.parse_args()
    
    if options.filename is not None:
        args.append(options.filename)
        
    if len(args)>1:
        print >> sys.stderr,  "arguments interpreted as FILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<1:
        parser.print_help()
        return
        
    h5_filename=args[0]

    fig = pylab.figure()
    show_it(fig,
            h5_filename,
            kalman_filename = options.kalman_filename,
            frame_start = options.start,
            frame_stop = options.stop,
            )
    pylab.show()

if __name__=='__main__':
    main()
