from __future__ import division
import numpy
from numpy import nan, pi
import tables as PT
import pytz # from http://pytz.sourceforge.net/
import datetime
import sys, os, sets
from optparse import OptionParser
import flydra.reconstruct
import flydra.analysis.result_utils as result_utils

def save_ascii_matrix(thefile,m):
    if hasattr(thefile,'write'):
        fd=thefile
    else:
        fd=open(thefile,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def do_it(filename,
          efilename,
          use_nth_observation=40,
          ):

    calib_dir = filename+'.recal'
    if not os.path.exists(calib_dir):
        os.makedirs(calib_dir)

    mylocals = {}
    myglobals = {}
    execfile(efilename,myglobals,mylocals)
    
    use_obj_ids = mylocals['long_ids']
    if 'bad' in mylocals:
        use_obj_ids = sets.Set(use_obj_ids)
        bad = sets.Set(mylocals['bad'])
        use_obj_ids = list(use_obj_ids.difference(bad))

    results = result_utils.get_results(filename,mode='r+')
    reconstructor = flydra.reconstruct.Reconstructor(results)
    
    camn2cam_id, cam_id2camns = result_utils.get_caminfo_dicts(results)
    
    cam_ids = cam_id2camns.keys()
    cam_ids.sort()
    
    kobs = results.root.kalman_observations

    data2d = results.root.data2d_distorted
    #use_idxs = numpy.arange(data2d.nrows)
    #frames = data2d.cols.frame[:]
    
    kobs_2d = results.root.kalman_observations_2d_idxs

    IdMat = []
    points = []
    for obj_id in use_obj_ids:
        print 'obj_id',obj_id
        k_use_idxs = kobs.getWhereList(
            kobs.cols.obj_id==obj_id,
            flavor='numpy' )
        obs_2d_idxs = kobs.readCoordinates( k_use_idxs,
                                            field='obs_2d_idx',
                                            flavor='numpy')
        kframes = kobs.readCoordinates( k_use_idxs,
                                        field='frame',
                                        flavor='numpy')
        kframes_use = kframes[::use_nth_observation]
        obs_2d_idxs_use = obs_2d_idxs[::use_nth_observation]
        npoints_by_cam_id = {}
        for cam_id in cam_ids:
            npoints_by_cam_id[cam_id] = 0
        
        for n_kframe, (kframe, obs_2d_idx) in enumerate(zip(kframes_use,obs_2d_idxs_use)):
            #print
            print 'kframe %d (%d of %d)'%(kframe,n_kframe,len(kframes_use))
            if 0:
                k_use_idx = k_use_idxs[n_kframe*use_nth_observation]
                print kobs.readCoordinates( numpy.array([k_use_idx]),
                                            flavor='numpy' )
            if PT.__version__ <= '1.3.3':
                obs_2d_idx_find = int(obs_2d_idx)
                kframe_find = int(kframe)
            else:
                obs_2d_idx_find = obs_2d_idx
                kframe_find = kframe
            obj_id_save = int(obj_id) # convert from possible numpy scalar

            #sys.stdout.write('  reading frame data...')
            #sys.stdout.flush()
            kobs_2d_data = kobs_2d.read( start=obs_2d_idx_find,
                                         stop=obs_2d_idx_find+1 )
            #sys.stdout.write('done\n')
            #sys.stdout.flush()

            assert len(kobs_2d_data)==1
            kobs_2d_data = kobs_2d_data[0]
            this_camns = kobs_2d_data[0::2]
            this_camn_idxs = kobs_2d_data[1::2]

            #sys.stdout.write('  doing frame selections...')
            #sys.stdout.flush()
            this_use_idxs = data2d.getWhereList( data2d.cols.frame==kframe_find,
                                                 flavor='numpy' )
            #sys.stdout.write('done\n')
            #sys.stdout.flush()
            
            if PT.__version__ <= '1.3.3':
                this_use_idxs = [int(t) for t in this_use_idxs]

            d2d = data2d.readCoordinates( this_use_idxs, flavor='numpy') # seems to require pytables >= 1.3.3
            #print 'd2d',d2d
            if len(this_camns) < 3:
                # not enough points to contribute to calibration
                continue

            n_pts = 0
            IdMat_row = []
            points_row = []
            for cam_id in cam_ids:
                found = False
                #print '    cam_id',cam_id
                for this_camn,this_camn_idx in zip(this_camns,this_camn_idxs):
                    #print '       %d %s'%(this_camn,camn2cam_id[this_camn])
                    if camn2cam_id[this_camn] != cam_id:
                        continue
                    npoints_by_cam_id[cam_id] = npoints_by_cam_id[cam_id] + 1
                    
                    this_camn_d2d = d2d[d2d.camn == this_camn]
                    #print '    this_camn_d2d',this_camn_d2d
                    for this_row in this_camn_d2d: # XXX could be sped up
                        if this_row.frame_pt_idx == this_camn_idx:
                            found = True
                            break
                if not found:
                    IdMat_row.append( 0 )
                    points_row.extend( [numpy.nan, numpy.nan, numpy.nan] )
                else:
                    n_pts += 1
                    IdMat_row.append( 1 )
                    points_row.extend( [this_row.x, this_row.y, 1.0] )
            #print 'IdMat_row',IdMat_row
            #print 'points_row',points_row
            IdMat.append( IdMat_row )
            points.append( points_row )
    IdMat = numpy.array(IdMat,dtype=numpy.uint8).T
    points = numpy.array(points,dtype=numpy.float32).T

    # resolution
    Res = []
    for cam_id in cam_ids:
        Res.append( reconstructor.get_resolution(cam_id) )
    Res = numpy.array( Res )

    cam_centers = numpy.asarray([reconstructor.get_camera_center(cam_id)[:,0]
                                 for cam_id in cam_ids])
    
    fd = open(os.path.join(calib_dir,'calibration_units.txt'),mode='w')
    fd.write(reconstructor.get_calibration_unit()+'\n')
    fd.close()
    
    results.close()

    save_ascii_matrix(os.path.join(calib_dir,'original_cam_centers.dat'),cam_centers)
    save_ascii_matrix(os.path.join(calib_dir,'IdMat.dat'),IdMat)
    save_ascii_matrix(os.path.join(calib_dir,'points.dat'),points)
    save_ascii_matrix(os.path.join(calib_dir,'Res.dat'),Res)
    
    fd = open(os.path.join(calib_dir,'camera_order.txt'),'w')
    for cam_id in cam_ids:
        fd.write('%s\n'%cam_id)
    fd.close()
    
    
    for cam_id in cam_ids:
        print 'cam_id %s: %d points'%(cam_id,npoints_by_cam_id[cam_id])
    
def main():
    usage = '%prog FILE EFILE [options]'
    
    parser = OptionParser(usage)
    
    (options, args) = parser.parse_args()
    
    if len(args)>2:
        print >> sys.stderr,  "arguments interpreted as FILE and EFILE supplied more than once"
        parser.print_help()
        return
    
    if len(args)<2:
        parser.print_help()
        return
        
    h5_filename=args[0]
    efilename = args[1]

    do_it(h5_filename,efilename)

if __name__=='__main__':
    main()
