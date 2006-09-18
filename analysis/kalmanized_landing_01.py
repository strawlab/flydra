from __future__ import division
from flydra.geom import LineSegment, ThreeTuple
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
R2D = 180/pi

class Distance(PT.IsDescription):
    obj_id = PT.Int32Col(pos=0,indexed=True)
    frame = PT.Int32Col(pos=1)
    distance = PT.FloatCol(pos=2)

def save_post_distances(kresults,post):
    
    postseg = LineSegment(ThreeTuple(post[0]),
                          ThreeTuple(post[1]))

    if hasattr( kresults.root,'post_distance'):
        print "removing old 'post_distance' table"
        kresults.removeNode( kresults.root.post_distance, recursive=True)
    print "creating 'post_distance' table"
    table = kresults.createTable(kresults.root,'post_distance',Distance,'distance to post')
    r = table.row

    whole_table = kresults.root.kalman_estimates.read(flavor='numpy')
    X = numpy.vstack((whole_table.field('x'),
                      whole_table.field('x'),
                      whole_table.field('z'))).T
    obj_ids = whole_table.field('obj_id')
    f = whole_table.field('frame')
    
    for i,(obj_id,f_i,x_i) in enumerate(zip(obj_ids,f,X)):
        if i%1000==0:
            print 'computing post distances (%d of %d)...'%(i,len(X))
        pt = ThreeTuple(x_i)
        dist = postseg.get_distance_from_point(pt)
        r['obj_id']=obj_id
        r['frame']=f_i
        r['distance']=dist
        r.append()
    table.flush()

def my_mag_2vecs(xd,yd):
    """normalize 2D vectors"""
    return numpy.sqrt( xd**2 + yd**2 )
    
def my_norm_2vecs(xd,yd):
    """normalize 2D vectors"""
    dir_vecs_mag = my_mag_2vecs(xd,yd)
    xnormed = xd/dir_vecs_mag
    ynormed = yd/dir_vecs_mag
    return xnormed, ynormed

def center_angles(angles):
    """put angles (in radians) in range -pi<angle<pi"""
    angles = numpy.ma.asarray(angles)
    return ((angles+pi)%(2*pi))-pi

def unwrap(angles):
    angles = numpy.ma.asarray(angles)
    adiff = angles[1:]-angles[:-1]
    adiff = center_angles(adiff)
    angles[1:] = angles[0]+numpy.cumsum(adiff)

def shade_angle_interval(ax,lower0=-90,upper0=90):
    orig_ylim = ax.get_ylim()
    ylim =  ax.dataLim.ymin(),ax.dataLim.ymax() #ax.get_ylim()

    lower_cycle_start = 360*round(ylim[0]/360.0)+lower0
    #lower_display = max(lower_cycle_start,ylim[0])
        
    upper_cycle_start = 360*round(ylim[1]/360.0)+upper0
    #upper_display = min(upper_cycle_start,ylim[1])

    diff = upper0-lower0
    cur_lower = lower_cycle_start
    while 1:
        cur_higher = cur_lower+diff
        #axhspan( max(cur_lower,ylim[0]), min(cur_higher,ylim[1]) )
        ax.axhspan( cur_lower, cur_higher,
                    facecolor='green',
                    edgecolor='green',
                    alpha=0.2)
        cur_lower += 360.0
        if cur_lower > ax.dataLim.ymax():
            break
    ax.set_ylim(orig_ylim)
    return orig_ylim
    
if 0:
    a = [0, pi/4, pi/2, pi, 3*pi/2]
    print a
    print center_angles( a )
    
if 1:
    if 0:
        # starved, small post
        filename = 'DATA20060828_181025.tracked_fixed_accel.h5'
        skip_obj_ids = range(110)

        # from ukine with recalibration
        post = [
            ( 429.1, 177.2, 190.5), # mm
            ( 427.6, 176.7, 199.5)]
        post = numpy.array(post)
        post = post*1e-3 # convert to meters
    elif 0:
        # sated, small post
        filename = 'DATA20060830_184701.tracked_fixed_accel.h5'
        skip_obj_ids = range(250)

        # from ukine with recalibration
        post = [
            ( 429.1, 177.2, 190.5), # mm
            ( 427.6, 176.7, 199.5)]
        post = numpy.array(post)
        post = post*1e-3 # convert to meters
    elif 1:
        # sated, tall post
        filename = 'DATA20060724_173517.tracked_fixed_accel.h5'
        skip_obj_ids = []
        
        # from ukine with recalibration
        post = [
            ( 464.3, 151.0, 84.2), #mm
            ( 454.1, 168.0, 246.3)]
        post = numpy.array(post)
        post = post*1e-3 # convert to meters

    post = numpy.array(post)
    postx=numpy.mean(post[:,0])
    posty=numpy.mean(post[:,1])
    postz0=post[0,2]
    postz1=post[1,2]
    kresults = PT.openFile(filename,mode="r+")
    if not hasattr( kresults.root,'post_distance'):
        save_post_distances(kresults,post)

    obj_ids = kresults.root.kalman_estimates.read(field='obj_id',flavor='numpy')

    if 1:
        #plot = None
        #plot = 'position_histogram'

        plot = 'traces_top_view'
        #plot = 'traces_side_view'
        #plot = 'traces_both_views'
        #plot = 'horiz_flight_direction'
        #plot = 'horiz_approach_angle'

        #plot = 'dist_vs_approach_angle'
        #plot = 'dist_vs_approach_angle_hist'
        #plot = 'dist_vs_approach_angle_hist_normalized'
        
        #plot = 'approach_angle_dur_hist'

        # ideas:
        # distance from post vs. approach angle (scatter plot)
        # angle from post vs. flight direction
        # angle from post vs. flight direction (as function of dist)

        plot_scale = 1000.0 # mm (1.0 = m, 1000.0 = mm)

        if plot is not None:
            import pylab
            #pylab.close('all')
            pylab.figure()
        my_xlim = (.1,.8)
        my_ylim = (0,.3)
        if plot == 'position_histogram':
            bin_size_m = .001
            xbins = numpy.arange(my_xlim[0],my_xlim[1],bin_size_m)
            ybins = numpy.arange(my_ylim[0],my_ylim[1],bin_size_m)
            hist2d = numpy.zeros( (ybins.shape[0], xbins.shape[0]))
        elif (plot == 'horiz_flight_direction'
              or plot == 'horiz_approach_angle'):
            bin_size_mm = 5
            xbins = numpy.arange(my_xlim[0],my_xlim[1],bin_size_mm)
            ybins = numpy.arange(my_ylim[0],my_ylim[1],bin_size_mm)
            accum = [[[] for j in xbins] for i in ybins]
            if plot == 'horiz_approach_angle':
                xdir_to_post = postx-xbins
                ydir_to_post = posty-ybins

                # i know there's a faster way using broadcasting...
                Xdir,Ydir = pylab.meshgrid(xdir_to_post,ydir_to_post)
                post_dir_x, post_dir_y = my_norm_2vecs(Xdir,Ydir)
                post_angles_rad = numpy.arctan2( post_dir_y, post_dir_x)
                assert post_angles_rad.shape == (len(ybins),len(xbins))
        elif plot in ['traces_top_view',
                      'traces_side_view',
                      'dist_vs_approach_angle',
                      'dist_vs_approach_angle_hist',
                      'dist_vs_approach_angle_hist_normalized',
                      ]:
            ax = pylab.subplot(1,1,1)
            if plot in ['dist_vs_approach_angle_hist',
                      'dist_vs_approach_angle_hist_normalized']:
                all_dist = []
                all_approach_angle = []
        elif plot in ['traces_both_views',
                      ]:
            ax1 = pylab.subplot(2,1,1)
            ax2 = pylab.subplot(2,1,2,sharex=ax1)
        elif plot in ['approach_angle_dur_hist',
                      ]:
            ax1 = pylab.subplot(3,1,1)
            ax2 = pylab.subplot(3,1,2,sharex=ax1,sharey=ax1)
            ax3 = pylab.subplot(3,1,3,sharex=ax1,sharey=ax1)
            all_approach_angle_close = []
            all_approach_angle_far = []
            all_approach_angle_farther = []
        else:
            print plot
            print plot in ['dist_vs_approach_angle_hist']
                      
            raise 'hmm'

        if 0:
            print 'WARNING: limiting analysis to first N traces'
            idxs = idxs[90:100]

        all_lens = []
        for obj_id in range( obj_ids.max()+1 ):
            if obj_id in skip_obj_ids:
                print 'skipping obj_id %d'%(obj_id,)
                continue
            row_idxs = numpy.nonzero( obj_ids == obj_id )[0]

            this_len = len(row_idxs)
            all_lens.append(this_len)

            if this_len < 10:
                print 'obj_id %d: %d frames, skipping'%(obj_id,this_len,)
                continue
            #print '%d frames'%(this_len,)

            xs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='x',flavor='numpy')
            ys = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='y',flavor='numpy')
            zs = kresults.root.kalman_estimates.readCoordinates(row_idxs,field='z',flavor='numpy')

            xdiff = xs[1:]-xs[:-1]
            ydiff = ys[1:]-ys[:-1]
            dir_x,dir_y = my_norm_2vecs(xdiff,ydiff)
            horiz_flight_angle_rad = numpy.arctan2( dir_y, dir_x )

##            # filter bad traces
##            if 1:
##                # sum(|accel|**2) 
##                X = numpy.asarray([xs,ys,zs]).transpose()
##                dt = 1/100.0 # framerate
##                d2Xdt2 = (X[2:]-2*X[1:-1]+X[:-2])/dt**2
##                sum_accel = numpy.sum( numpy.sqrt(numpy.sum( d2Xdt2**2,axis=1) ))
##                N = stop-start
##                mean_accel = sum_accel/N
##                #print mean_accel
##                if mean_accel > 2000:
##                    print 'WARNING: skipping trace %d-%d (frames %d-%d) because too jerky'%(
##                        start,stop,start_frame,stop_frame)
##                    continue
                
##                # trajectory angle criterion
##                ang_diff=horiz_flight_angle_rad[1:]-horiz_flight_angle_rad[:-1]
##                ang_diff = center_angles(ang_diff)
##                mean_ang_diff = numpy.sum(abs(ang_diff))/len(ang_diff)
##                #print mean_ang_diff
##                if mean_ang_diff > 0.8:
##                    print 'WARNING: skipping trace %d-%d (frames %d-%d) because too turny'%(
##                        start,stop,start_frame,stop_frame)
##                    continue
            
            if plot == 'position_histogram':
                xidx = xbins.searchsorted(xs)
                yidx = ybins.searchsorted(ys)
                
                for i,j in zip(yidx,xidx):
                    hist2d[i,j]+=1
            elif plot == 'traces_top_view':
                line, = pylab.plot(xs*plot_scale,ys*plot_scale,'b-') # plot in mm
                #pylab.text( xs[0]*plot_scale, ys[0]*plot_scale, str(obj_id) )
                line.set_alpha(0.2)
            elif plot == 'traces_side_view':
                if 1:
                    #ax.text(xs[0],zs[0],str(start))
                    #pylab.plot(xs,zs,'bo')
                    pylab.plot(xs,zs)
                    
                if 0:
                    xs2 = ccf.get_X()[start:stop,0]
                    zs2 = ccf.get_X()[start:stop,2]
                    pylab.plot(xs2,zs2,'r.')
            elif plot == 'traces_both_views':
                if 1:
                    ax1.plot(xs,ys)
                    ax2.plot(xs,zs)
                    if 1:
                        ax1.plot(xs[::10],ys[::10],'b.',markersize=6)
                        ax2.plot(xs[::10],zs[::10],'b.',markersize=6)
                elif 0:
                    ax1.plot(xs,ys,'.')
                    ax2.plot(xs,zs,'.')
                elif 0:
                    interv = 5
                    print xs[::interv], ys[::interv]
                    ax1.plot(xs[::interv],ys[::interv],'.')
                    ax2.plot(xs[::interv],zs[::interv],'.')
            elif plot in ['dist_vs_approach_angle',
                          'dist_vs_approach_angle_hist',
                          'dist_vs_approach_angle_hist_normalized',
                          'approach_angle_dur_hist',
                          ]:

                # 1) calculate distance
                #  a) make same length as approach angle vector:
                xav = (xs[1:]+xs[:-1])*0.5
                yav = (ys[1:]+ys[:-1])*0.5
                #  b) find component distance
                xdist = xav-postx
                ydist = yav-posty
                #  c) find distance
                dist = my_mag_2vecs(xdist,ydist)

                # 2) calculate approach angle
                #  a) calculate horizontal flight direction angle

                # b) calculate post angle
                xdir_to_post = postx-xav
                ydir_to_post = posty-yav
                dir_x,dir_y = my_norm_2vecs(xdir_to_post,ydir_to_post)
                post_angle_rad = numpy.arctan2(dir_y,dir_x)

                # c) approach angle
                approach_angle = center_angles(post_angle_rad-horiz_flight_angle_rad)
                if plot=='dist_vs_approach_angle':
                    #pylab.plot(dist,approach_angle*R2D,'.')

                    if 1:
                        unwrap(approach_angle)
                    pylab.plot(dist,approach_angle*R2D)
                    pylab.plot(dist[::10],(approach_angle*R2D)[::10],'b.',markersize=6)
                    #ax.text(dist[0],approach_angle[0]*R2D,str(start))
                    #ax.text(dist[-1],approach_angle[-1]*R2D,str(start))
                    
                    pylab.xlabel('horizontal distance (mm)')
                    pylab.ylabel('approach angle (deg)')
                                        
                elif plot in ['dist_vs_approach_angle_hist',
                              'dist_vs_approach_angle_hist_normalized']:
                    all_dist.append( dist )
                    all_approach_angle.append( approach_angle )
                elif plot in ['approach_angle_dur_hist']:
                    close_condition = dist < 50
                    far_condition = (dist >= 50) & (dist < 100)
                    farther_condition = (dist >= 100)
                    close_idx = numpy.nonzero(close_condition)
                    far_idx = numpy.nonzero(far_condition)
                    farther_idx = numpy.nonzero(farther_condition)

                    all_approach_angle_close.append(approach_angle[close_idx])
                    all_approach_angle_far.append(approach_angle[far_idx])
                    all_approach_angle_farther.append(approach_angle[farther_idx])
            elif (plot == 'horiz_flight_direction' or
                  plot == 'horiz_approach_angle'):
            
                xav = (xs[1:]+xs[:-1])*0.5
                yav = (ys[1:]+ys[:-1])*0.5

                xidx = xbins.searchsorted(xav)
                yidx = ybins.searchsorted(yav)
                for idx in range(len(xav)):
                    i = yidx[idx]; j = xidx[idx]
                    accum[i][j].append( (dir_x[idx],
                                         dir_y[idx]) )
                    
        all_lens = numpy.array(all_lens)
        sort_idxs = numpy.argsort(all_lens)
        for si in sort_idxs[-10:]:
            long_len = all_lens[ si ]
            print 'long trace: %d frames'%(long_len,)
        
        if plot in ['traces_top_view']:
            pylab.title(filename)
            pylab.plot([postx*plot_scale],[posty*plot_scale],'ko')
            ax.set_aspect( 'equal', adjustable='datalim' )
            if plot_scale == 1000.0:
                pylab.xlabel('X (mm)')
                pylab.ylabel('Y (mm)')
            elif plot_scale == 1.0:
                pylab.xlabel('X (m)')
                pylab.ylabel('Y (m)')
        elif plot == 'traces_side_view':
            pylab.plot([postx,postx],
                       [postz0,postz1],'k-',lw=5)
            ax.set_aspect( 'equal', adjustable='datalim' )
            pylab.xlabel('X (mm)')
            pylab.ylabel('Z (mm)')
        elif plot == 'traces_both_views':
            ax1.plot([postx],[posty],'ko')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')
            ax1.set_aspect( 'equal', adjustable='datalim' )
            
            ax2.plot([postx,postx],
                     [postz0,postz1],'k-',lw=5)
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Z (mm)')
            ax2.set_aspect( 'equal', adjustable='datalim' )
            
        elif plot == 'position_histogram':
            X,Y = pylab.meshgrid(xbins,ybins)
            maxval = hist2d.max()
            print 'maxval',maxval
            meanval = numpy.mean( numpy.mean( hist2d ))
            print 'meanval', meanval
            
            pylab.pcolor(X,Y,hist2d,
                         vmin=0,
                         vmax=3*meanval,
                         shading='flat')
            
            pylab.plot([postx],[posty],'wo')
            pylab.title(filename)
            
        elif plot == 'dist_vs_approach_angle':
            ylim = shade_angle_interval(ax,lower0=-10,upper0=10)
            shade_angle_interval(ax,lower0=-90,upper0=90)

            ylabel_start = 90*round(ylim[0]/90.0)
            ylabel_stop =  90*round(ylim[1]/90.0)
            yticks = [ angle for angle in range(ylabel_start,ylabel_stop+1,90)]
            yticklabels = [ repr((angle+180)%360.0-180) for angle in yticks ]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            
        elif plot in ['dist_vs_approach_angle_hist',
                      'dist_vs_approach_angle_hist_normalized']:
            all_dist = numpy.hstack(all_dist)
            all_approach_angle = numpy.hstack(all_approach_angle)
            dist_bin_size = 2.0 # mm
            dist_bins = numpy.arange(0.0,
                                     all_dist.max()+dist_bin_size,
                                     dist_bin_size )
            angle_bins = numpy.linspace(-pi,pi,100)
            hist2d = numpy.zeros(
                (angle_bins.shape[0], dist_bins.shape[0]),
                dtype=numpy.Float64)
            dist_idx = dist_bins.searchsorted(all_dist)
            angle_idx = angle_bins.searchsorted(all_approach_angle)
            for idx in range(len(all_dist)):
                i = angle_idx[idx]; j = dist_idx[idx]
                try:
                    hist2d[i,j]+=1.0
                except IndexError,err:
                    print i,j
                    print hist2d.shape
                    raise

            if plot == 'dist_vs_approach_angle_hist_normalized':
                n_dists = numpy.sum(hist2d,axis=0)
                hist2d = hist2d/n_dists[numpy.NewAxis,:] # normalize probability
                hist2d = numpy.ma.masked_where( numpy.isnan(hist2d), hist2d )
            DIST,ANGLE = pylab.meshgrid(dist_bins,angle_bins)
            pylab.pcolor(DIST,ANGLE*R2D,hist2d,shading='flat')
            pylab.xlabel('horizontal distance (mm)')
            pylab.ylabel('approach angle (deg)')

            ax = pylab.gca()
            if 1:
                ylim = [-180,180]
                ax.set_ylim(ylim)

                xlim = [dist_bins[0],dist_bins[-1]]
                ax.set_xlim(xlim)
            else:
                ylim = ax.get_ylim()
            ylabel_start = 90*round(ylim[0]/90.0)
            ylabel_stop =  90*round(ylim[1]/90.0)
            yticks = [ angle for angle in range(ylabel_start,ylabel_stop+1,90)]
            #yticklabels = [ repr((angle+180)%360.0-180) for angle in yticks ]
            yticklabels = [ repr(angle) for angle in yticks ]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        elif plot in ['approach_angle_dur_hist']:
            angle_bins = numpy.linspace(-pi,pi,100)*R2D
            normed = True
            
            all_approach_angle_close = numpy.hstack(all_approach_angle_close)
            ax1.hist(all_approach_angle_close*R2D,
                     bins=angle_bins,
                     normed=normed,
                     )
            pylab.ylabel('probability')
            pylab.xlabel('approach angle (deg)')

            all_approach_angle_far = numpy.hstack(all_approach_angle_far)
            ax2.hist(all_approach_angle_far*R2D,
                     bins=angle_bins,
                     normed=normed,
                     )
            pylab.ylabel('probability')
            pylab.xlabel('approach angle (deg)')

            all_approach_angle_farther = numpy.hstack(
                all_approach_angle_farther)
            ax3.hist(all_approach_angle_farther*R2D,
                     bins=angle_bins,
                     normed=normed,
                     )
            pylab.ylabel('probability')
            pylab.xlabel('approach angle (deg)')

            
        elif (plot == 'horiz_flight_direction' or
              plot == 'horiz_approach_angle'):
            horiz_flight_angles_rad = nan*numpy.ones(
                (ybins.shape[0], xbins.shape[0]),
                dtype=numpy.Float64)
            horiz_flight_r = nan*numpy.ones(
                (ybins.shape[0], xbins.shape[0]),
                dtype=numpy.Float64)
            for i in range(ybins.shape[0]):
                for j in range(xbins.shape[0]):
                    vals = accum[i][j]
                    if len(vals):
                        vals = numpy.asarray(vals)
                        mean_x = numpy.mean( vals[:,0] )
                        mean_y = numpy.mean( vals[:,1] )
                        horiz_flight_angles_rad[i,j] = numpy.arctan2( mean_y, mean_x )
                        horiz_flight_r[i,j] = numpy.sqrt(mean_x**2 + mean_y**2)
                        if 0:
                            print accum[i][j]
                            print vals
                            print i,j
                            print horiz_flight_angles_rad[i,j]
                            print
            X,Y = pylab.meshgrid(xbins,ybins)
            if (plot == 'horiz_flight_direction' or
                plot == 'horiz_approach_angle'):
                ax1 = pylab.subplot(2,1,1)
                ax2 = pylab.subplot(2,1,2)
                
                cond = numpy.isnan(horiz_flight_r)
                
                horiz_flight_angles_rad = numpy.ma.masked_where(
                    cond, horiz_flight_angles_rad )
                
                horiz_flight_r = numpy.ma.masked_where(
                    cond, horiz_flight_r )
                
                if plot == 'horiz_flight_direction':
                    ax1.pcolor(X,Y,horiz_flight_angles_rad*R2D,shading='flat')
                    pylab.axis('equal')
                    ax2.pcolor(X,Y,horiz_flight_r,shading='flat')
                    pylab.axis('equal')
                elif plot == 'horiz_approach_angle':
                    #diff_angles_rad = numpy.ma.masked_array(post_angles_rad)-horiz_flight_angles_rad
                    diff_angles_rad = post_angles_rad-horiz_flight_angles_rad
                    diff_angles_rad = center_angles(diff_angles_rad)
                    ax1.pcolor(X,Y,diff_angles_rad*R2D,shading='flat')
                    pylab.axis('equal')
                    ax2.pcolor(X,Y,horiz_flight_r,shading='flat')
                    pylab.axis('equal')                    
            pylab.plot([postx],[posty],'wo')
        #pylab.axis('equal')
        if plot is not None:
            #pylab.savefig('topview.png')
            pylab.show()
