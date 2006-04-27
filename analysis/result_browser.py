#!/usr/bin/env python
import time, StringIO, sets, sys
import math, os
import numpy as nx
import numpy as mlab
import tables as PT
print 'using pytables',PT.__version__
print '  from',PT.__file__
import matplotlib
import pylab
import matplotlib.ticker
from pylab import figure, plot, clf, imshow, cm, setp, figtext
from pylab import gca, title, axes, ion, ioff, gcf, savefig
from matplotlib.ticker import LinearLocator
import PQmath
import cgtypes
import threading

import flydra.undistort as undistort
import FlyMovieFormat

from numarray.ieeespecial import getnan, nan, inf
import numarray.linear_algebra

try:
    import Pyro.core, Pyro.errors
except ImportError,x:
    print 'skipping Pyro import:',x
else:
    Pyro.core.initClient(banner=0)

PROXY_PYRO = False

class ExactMovieInfo(PT.IsDescription):
    cam_id             = PT.StringCol(16,pos=0)
    filename           = PT.StringCol(255,pos=1)
    start_frame        = PT.Int32Col(pos=2)
    stop_frame         = PT.Int32Col(pos=3)
    start_timestamp    = PT.FloatCol(pos=4)
    stop_timestamp     = PT.FloatCol(pos=5)

class IgnoreFrames(PT.IsDescription):
    start_frame        = PT.Int32Col(pos=0)
    stop_frame         = PT.Int32Col(pos=1)

class SmoothData(PT.IsDescription):
    frame = PT.Int32Col(pos=0)
    x     = PT.FloatCol(pos=1)
    y     = PT.FloatCol(pos=2)
    z     = PT.FloatCol(pos=3)
    qw    = PT.FloatCol(pos=4)
    qx    = PT.FloatCol(pos=5)
    qy    = PT.FloatCol(pos=6)
    qz    = PT.FloatCol(pos=7)

class TimedVectors(PT.IsDescription):
    frame = PT.Int32Col(pos=0)
    fx    = PT.FloatCol(pos=1)
    fy    = PT.FloatCol(pos=2)
    fz    = PT.FloatCol(pos=3)
    
def make_new_fmt(results):
    """convert all 2D data into camera-by-camera tables"""
    Info2D = results.root.data2d.description # 2D data format for PyTables
    if hasattr(results.root, 'cam_by_cam_2d'):
        return # already made
    # (can delete with results.root.cam_by_cam_2d._f_remove() )
    # or results.removeNode( results.root.cam_by_cam_2d, recursive=True)
    status("making new cam_by_cam_2d tables...")
    cam_by_cam_2d = results.createGroup( results.root, 'cam_by_cam_2d' )
    
    tables_by_camn = {}
    for oldrow in results.root.data2d:
        camn = oldrow['camn']
        if camn not in tables_by_camn:
            table_name = 'camn'+str(camn)
            tables_by_camn[camn] = results.createTable( cam_by_cam_2d, table_name,
                                                        Info2D, "2d data" )
        table = tables_by_camn[camn]
        newrow = table.row
        for attr in Info2D._v_names:
            newrow[attr] = oldrow[attr] #copy all attributes
        newrow.append()
        
    for camn,table in tables_by_camn.iteritems():
        table.flush()
    status("done")

def status(status_string):
    print " status:",status_string
    sys.stdout.flush()

def my_subplot(n):
    x_space = 0.05
    y_space = 0.15
    
    left = n*0.2 + x_space
    bottom = 0 + + y_space
    w = 0.2 - (x_space*1.5)
    h = 1.0 - (y_space*2)
    return axes([left,bottom,w,h])

def dougs_subplot(n,n_rows=2,n_cols=3):
    # 2 rows and n_cols
    
    rrow = n / n_cols # reverse row
    row = n_rows-rrow-1 # row number
    col = n % n_cols
    
    x_space = (0.02/n_cols)
    y_space = 0.0125

    y_size = 0.48
    
    left = col*(1.0/n_cols) + x_space
    bottom = row*y_size + y_space
    w = (1.0/n_cols) - x_space
    h = y_size - 2*y_space
    return axes([left,bottom,w,h])

proxy_spawner = None
def get_server(cam_id,port=9888):
    if PROXY_PYRO:
        import urlparse, socket
        global proxy_spawner

        if proxy_spawner is None:
            hostname = 'localhost' # requires tunnelling (e.g. over ssh)

            proxy_URI = "PYROLOC://%s:%d/%s" % (hostname,port,'proxy_spawner')
            print 'connecting to',proxy_URI,'...',
            proxy_spawner = Pyro.core.getProxyForURI(proxy_URI)

        # make sure URI is local (so we can forward through tunnel)
        URI = str(proxy_spawner.spawn_proxy(cam_id))
        URI=URI.replace('PYRO','http') # urlparse chokes on PYRO://
        URIlist = list( urlparse.urlsplit(str(URI)) )
        network_location = URIlist[1]
        localhost = socket.gethostbyname(socket.gethostname())
        port = network_location.split(':')[1]
        URIlist[1] = '%s:%s'%(localhost,port)
        URI = urlparse.urlunsplit(URIlist)
        URI=URI.replace('http','PYRO')
    else:
        hostname = cam_id.split(':')[0]

        URI = "PYROLOC://%s:%d/%s" % (hostname,port,'frame_server')

    frame_server = Pyro.core.getProxyForURI(URI)
    frame_server.noop()
        
    return frame_server

def get_camn_and_timestamp(results, cam, frame):

    if isinstance(cam,str):
        cam_id = cam
        possible_camns = []
        for row in results.root.cam_info:
            if row['cam_id'] == cam_id:
                possible_camns.append( row['camn'] )
    else:
        camn = cam
        possible_camns = [camn]
        
    found = False
    for row in results.root.data2d:
        if row['frame'] == frame:
            camn = row['camn']
            if camn in possible_camns:
                timestamp = row['timestamp']
                found = True
                break
    if not found:
        raise ValueError("No data found for cam_id and frame")
    return camn, timestamp
        
def flip_line_direction(results,frame,typ='best'):
    if typ=='best':
        data3d = results.root.data3d_best
    elif typ=='fastest':
        data3d = results.root.data3d_fastest

    nrow = None
    for row in data3d.where( data3d.cols.frame == frame ):
        nrow = row.nrow()
        p2, p4, p5 = row['p2'], row['p4'], row['p5']
    if nrow is None:
        raise ValueError('could not find frame')
    nrow = int(nrow) # hmm weird pytables bug?
    data3d.cols.p2[nrow] = -p2
    data3d.cols.p4[nrow] = -p4
    data3d.cols.p5[nrow] = -p5

def my_normalize(V):
    v = nx.asarray(V)
    assert len(v.shape)==1
    u = v/ math.sqrt( nx.sum( v**2) )
    return u

def sort_on_col0( a, b ):
    a0 = a[0]
    b0 = b[0]
    if a0 < b0: return -1
    elif a0 > b0: return 1
    else: return 0

def auto_flip_line_direction(results,start_frame,stop_frame,typ='best',
                             skip_allowance = 5,                             
                             ):
    if typ=='best':
        data3d = results.root.data3d_best
    elif typ=='fastest':
        data3d = results.root.data3d_fastest

    assert stop_frame-start_frame > 1
    
    frame_and_dir_list = [ (row['frame'], -row['p2'],row['p4'],-row['p5']) for row in data3d.where( start_frame <= data3d.cols.frame <= stop_frame ) ]
    frame_and_dir_list.sort(sort_on_col0)
    frame_and_dir_list = nx.array( frame_and_dir_list )
    bad_idx=list(getnan(frame_and_dir_list[:,1])[0])
    good_idx = [i for i in range(len(frame_and_dir_list[:,1])) if i not in bad_idx]
    frame_and_dir_list = [ frame_and_dir_list[i] for i in good_idx]
    
    prev_frame = frame_and_dir_list[0][0]
    prev_dir = my_normalize(frame_and_dir_list[0][1:4])
    
    cos_90 = math.cos(math.pi/4)
    frames_flipped = []
    
    for frame_and_dir in frame_and_dir_list[1:]:
        this_frame = frame_and_dir[0]
        this_dir = my_normalize(frame_and_dir[1:4])

        try:
            cos_theta = nx.dot(this_dir, prev_dir)
        except Exception,x:
            print 'exception this_dir prev_dir',this_dir,prev_dir
            raise
        except:
            print 'hmm'
            raise
        theta_deg = math.acos(cos_theta)/math.pi*180
        
        if this_frame in [188305, 188306, 188307]:
            print '*'*10,theta_deg
        
#        print this_frame, this_dir, cos_theta, math.acos(cos_theta)/math.pi*180
#        print
        
        dt_frames = this_frame - prev_frame
        
        prev_frame = this_frame
        prev_dir = this_dir

        
        if dt_frames > skip_allowance:
            print 'frame %d skipped because previous %d frames not found'%(this_frame,skip_allowance)
            continue
        if theta_deg > 90:
            flip_line_direction(results,this_frame,typ=typ)
            prev_dir = -prev_dir
            frames_flipped.append(this_frame)
    return frames_flipped

def auto_flip_line_direction_hang(results,start_frame,stop_frame,typ='best',
                              ):
    if typ=='best':
        data3d = results.root.data3d_best
    elif typ=='fastest':
        data3d = results.root.data3d_fastest

    assert stop_frame-start_frame > 1
    
    frame_and_dir_list = [ (row['frame'], -row['p2'],row['p4'],-row['p5']) for row in data3d.where( start_frame <= data3d.cols.frame <= stop_frame ) ]
    frame_and_dir_list.sort(sort_on_col0)
    frame_and_dir_list = nx.array( frame_and_dir_list )
    bad_idx=list(getnan(frame_and_dir_list[:,1])[0])
    good_idx = [i for i in range(len(frame_and_dir_list[:,1])) if i not in bad_idx]
    frame_and_dir_list = [ frame_and_dir_list[i] for i in good_idx]
    
    frames_flipped = []
    
    for frame_and_dir in frame_and_dir_list:
        this_frame = frame_and_dir[0]
        this_dir = my_normalize(frame_and_dir[1:4])
        if this_dir[2] < -0.1:
            flip_line_direction(results,this_frame,typ=typ)
            frames_flipped.append(this_frame)
    return frames_flipped

def time2frame(results,time_double,typ='best'):
    assert isinstance(time_double,float)

    if typ=='best':
        table = results.root.data3d_best
    elif typ=='fastest':
        table = results.root.data3d_fastest

    status('copying column to Python')
    find3d_time = nx.array( table.cols.find3d_time )
    status('searching for closest time')
    idx = nx.argmin(nx.abs(find3d_time-time_double))
    status('found index %d'%idx)
    return table.cols.frame[idx]

def from_table_by_frame(table,frame,colnames=None):
    if colnames is None:
        colnames='x','y'

    def values_for_keys(dikt,keys):
        return [dikt[key] for key in keys]
    
    rows = [values_for_keys(x,colnames) for x in table if x['frame']==frame]
    #rows = [values_for_keys(x,colnames) for x in table.where( table.cols.frame==frame)]
    return rows

def get_pmat(results,cam_id):
    return nx.array(results.root.calibration.pmat.__getattr__(cam_id))

def get_intlin(results,cam_id):
    return nx.array(results.root.calibration.intrinsic_linear.__getattr__(cam_id))

def get_intnonlin(results,cam_id):
    return nx.array(results.root.calibration.intrinsic_nonlinear.__getattr__(cam_id))

def get_resolution(results,cam_id):
    return tuple(results.root.calibration.resolution.__getattr__(cam_id))

def get_frames_with_3d(results):
    table = results.root.data3d_best
    return [x['frame'] for x in table]

def redo_3d_calc(results,frame,reconstructor=None,verify=True,overwrite=False):
    import flydra.reconstruct
    
    data3d = results.root.data3d_best
    data2d = results.root.data2d
    cam_info = results.root.cam_info
    
    Xorig, camns_used, nrow = [ ((x['x'],x['y'],x['z']),x['camns_used'],x.nrow())
##                                for x in data3d.where( data3d.cols.frame==frame )][0]
                                for x in data3d if x['frame']==frame ][0]
    nrowi=int(nrow)
    assert nrowi==nrow
    camns_used = map(int,camns_used.split())

    status('testing frame %d with cameras %s'%(frame,str(camns_used)))

    if reconstructor is None:
        reconstructor = flydra.reconstruct.Reconstructor(results)

    cam_ids_and_points2d = []
    for camn in camns_used:
        cam_id = [row['cam_id'] for row in cam_info if row['camn']==camn][0]
        value_tuple_list = [(row['x'],row['y'],row['area'],
                             row['slope'],row['eccentricity'],
                             row['p1'],row['p2'],row['p3'],row['p4'])
                            for row in data2d if row['frame']==frame and row['camn']==camn]
        if len(value_tuple_list) == 0:
            raise RuntimeError('no 2D data for camn %d frame %d'%(camn,frame))
        assert len(value_tuple_list) == 1
        value_tuple = value_tuple_list[0]
##        value_tuple = [(row['x'],row['y'],row['area'],
##                        row['slope'],row['eccentricity'],
##                        row['p1'],row['p2'],row['p3'],row['p4'])
##                       ##                       for row in data2d.where( data2d.cols.frame==frame )
####                       if row['camn']==camn][0]
##                       for row in data2d if row['frame']==frame and row['camn']==camn][0]
        x,y,area,slope,eccentricity,p1,p2,p3,p4 = value_tuple
        cam_ids_and_points2d.append( (cam_id, value_tuple) )
        
    Xnew, Lcoords = reconstructor.find3d(cam_ids_and_points2d)
    if verify:
        assert nx.allclose(Xnew,Xorig)
    if overwrite:
        raise NotImplementedError("not done yet")

def get_3d_frame_range_with_2d_info(results):
    data3d = results.root.data3d_best
    data2d = results.root.data2d

    frames_3d = [ x['frame'] for x in data3d ]
    frames_3d.sort()
    frame_min = frames_3d[0]
    frame_max = frames_3d[-1]
    return frame_min, frame_max

def get_caminfo_dicts(results):
    # camera info
    cam_info = results.root.cam_info
    cam_id2camns = {}
    camn2cam_id = {}
    
    for row in cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        cam_id2camns.setdefault(cam_id,[]).append(camn)
        camn2cam_id[camn]=cam_id
    return camn2cam_id, cam_id2camns

def summarize(results):
    res = StringIO.StringIO()

    camn2cam_id, cam_id2camns = get_caminfo_dicts(results)
##    # camera info
##    cam_info = results.root.cam_info
##    cam_id2camns = {}
##    camn2cam_id = {}
    
##    for row in cam_info:
##        cam_id, camn = row['cam_id'], row['camn']
##        cam_id2camns.setdefault(cam_id,[]).append(camn)
##        camn2cam_id[camn]=cam_id

    # 2d data
    data2d = results.root.data2d

    n_2d_rows = {}
    for camn in camn2cam_id.keys():
##        n_2d_rows[camn] = len( [ x for x in data2d.where( data2d.cols.camn == camn ) ])
        n_2d_rows[camn] = len( [ x for x in data2d if x['camn'] == camn ])

    #print >> res, 'camn cam_id n_2d_rows'
    for camn in camn2cam_id.keys():
        print >> res, "camn %d ('%s') %d frames of 2D info"%(camn, camn2cam_id[camn], n_2d_rows[camn])

    print >> res

    # 3d data
    data3d = results.root.data3d_best
    print >> res, len(data3d),'frames of 3d information'
    frames = list(data3d.cols.frame)
    frames.sort()
    print >> res, '  (start: %d, stop: %d)'%(frames[0],frames[-1])
    

    frame2camns_used = {}
    for row in data3d:
        camns_used = map(int,row['camns_used'].split())
        if len(getnan(row['p0'])[0]):
            orient_info = False
        else:
            orient_info = True
        frame2camns_used[row['frame']]=camns_used, orient_info
    
    nframes_by_n_camns_used = {}
    nframes2_by_n_camns_used = {} # with orient_info
    for f, (camns_used, orient_info) in frame2camns_used.iteritems():
        nframes_by_n_camns_used[len(camns_used)]=nframes_by_n_camns_used.get(len(camns_used),0) + 1
        if orient_info:
            nframes2_by_n_camns_used[len(camns_used)]=nframes2_by_n_camns_used.get(len(camns_used),0) + 1
        else:
            nframes2_by_n_camns_used.setdefault(len(camns_used),0)
        
        #orig_value = nframes_by_n_camns_used.setdefault( len(camns_used), 0).add(1)
        
    for n_camns_used,n_frames in nframes_by_n_camns_used.iteritems():
        print >> res, ' with %d camns: %d frames (%d with orientation)'%(n_camns_used, n_frames, nframes2_by_n_camns_used[n_camns_used] )
    
    res.seek(0)
    return res.read()

def get_f_xyz_L_err( results, max_err = 10, typ = 'best', include_timestamps=False):
    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best

    if max_err is not None:
        f = []
        x = []
        y = []
        z = []
        xyz=[]
        L = []
        err = []
        timestamps = []
        for row in data3d.where( data3d.cols.mean_dist <= max_err ):
            f.append( row['frame'] )
            xyz.append( (row['x'],row['y'],row['z']) )
            L.append( (row['p0'],row['p1'],
                       row['p2'],row['p3'],
                       row['p4'],row['p5']))
            err.append( row['mean_dist'] )
            timestamps.append( row['timestamp'] )
        f = nx.array(f)
        xyz = nx.array(xyz)
        L = nx.array(L)
        err = nx.array(err)
        timestamps = nx.array(timestamps)
    else:
        frame_col = data3d.cols.frame
        if not len(frame_col):
            print 'no 3D data'
            return
        f = nx.array(frame_col)
        timestamps = nx.array(timestamps)
        x = nx.array(data3d.cols.x)
        y = nx.array(data3d.cols.y)
        z = nx.array(data3d.cols.z)
        xyz = nx.concatenate( (x[:,nx.NewAxis],
                               y[:,nx.NewAxis],
                               z[:,nx.NewAxis]),
                              axis = 1)
        p0 = nx.array(data3d.cols.p0)[:,nx.NewAxis]
        p1 = nx.array(data3d.cols.p1)[:,nx.NewAxis]
        p2 = nx.array(data3d.cols.p2)[:,nx.NewAxis]
        p3 = nx.array(data3d.cols.p3)[:,nx.NewAxis]
        p4 = nx.array(data3d.cols.p4)[:,nx.NewAxis]
        p5 = nx.array(data3d.cols.p5)[:,nx.NewAxis]
        L = nx.concatenate( (p0,p1,p2,p3,p4,p5), axis=1 )
        err = nx.array(data3d.cols.mean_dist)

    if hasattr(results.root,'ignore_frames'):
        good = nx.argsort(f)
        good_set = sets.Set( nx.argsort( f ) )
        for row in results.root.ignore_frames:
            start_frame, stop_frame = row['start_frame'], row['stop_frame']
            head = nx.where( f < start_frame )
            tail = nx.where( f > stop_frame )
            head_set = sets.Set(head[0])
            tail_set = sets.Set(tail[0])

            good_set = (good_set & head_set) | (good_set & tail_set)
        good_idx = list( good_set )
        good_idx.sort()
    else:
        good_idx = nx.argsort(f)
        
    f = nx.take(f,good_idx)
    xyz = nx.take(xyz,good_idx)
    L = nx.take(L,good_idx)
    err = nx.take(err,good_idx)
    timestamps = nx.take(timestamps,good_idx)

    rval = [f,xyz,L,err]
    if include_timestamps:
        rval.append( timestamps )
    return tuple(rval)

def plot_movie_2d(results,
                  start_idx=0,
                  stop_idx=-1,
                  fstart=None,
                  fstop=None,
                  typ='best',
                  show_err=False,
                  max_err=10,
                  onefigure=True,
                  show_cam_usage=False):
    if not hasattr(results.root, 'cam_by_cam_2d'):
        make_new_fmt(results)
    ioff()
    try:
        import flydra.reconstruct
        reconstructor = flydra.reconstruct.Reconstructor(results)
        
        data3d = results.root.data3d_best
        
        # get 3D data
        f,X,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)
        if fstart is not None:
            fstart = max(f[start_idx],fstart)
            if fstart != f[start_idx]:
                start_idx = numarray.nonzero(f>=fstart)[0][0]
        else:
            fstart = f[start_idx]
        if fstop is not None:
            fstop = min(f[stop_idx],fstop)
            if fstop != f[stop_idx]:
                stop_idx = numarray.nonzero(f<=fstop)[0][-1]
        else:
            fstop = f[stop_idx]
        f=f[start_idx:stop_idx]
        X=X[start_idx:stop_idx]
        X = nx.concatenate( (X, nx.ones( (X.shape[0],1) )), axis=1 )
        X.transpose()

        # get camera information
        camns = [ row['camn'] for row in results.root.cam_info]
        camn2cam_id = {}
        cam_id2camn = {}
        cam_ids_unique = True
        for row in results.root.cam_info:
            cam_id = row['cam_id']
            camn = row['camn']
            camn2cam_id[ camn] = cam_id
            if cam_id in cam_id2camn:
                cam_ids_unique = False
            cam_id2camn[ cam_id ] = camn
        cam_ids = [ camn2cam_id[camn] for camn in camns ]
        cam_ids.sort()
        if onefigure:
            # plots from the bottom up
            cam_ids.reverse()
        if cam_ids_unique:
            # can order by cam_id
            camns = [ cam_id2camn[cam_id] for cam_id in cam_ids ]

        if onefigure:
            ncams = len(camns)
            height = 0.8/ncams
        ax = None
        useful_axes = {}
        for i, camn in enumerate(camns):
            cam_id = camn2cam_id[camn]

            # project 3D data through camera calibration to get 2D reprojection
            xy=reconstructor.find2d(cam_id,X)

            # get original 2D points
            f2 = []
            x2 = []
            y2 = []
            p1 = []
            camn_used = []

            table_name = 'camn'+str(camn)
            table = getattr(results.root.cam_by_cam_2d,table_name)
            for row in table.where( fstart <= table.cols.frame <= fstop ):
                frame = row['frame']
                f2.append(frame)
                x2.append(row['x'])
                y2.append(row['y'])
                p1.append(row['p1'])
                if show_cam_usage:
                    cu = nan
                    for row3d in data3d.where( frame == data3d.cols.frame ):
                        if str(camn) in row3d['camns_used']:
                            cu = 1
                    camn_used.append( cu )
                    
            f2 = nx.array(f2)
            x2 = nx.array(x2)
            y2 = nx.array(y2)
            p1 = nx.array(p1)
            p1[~nx.isnan(p1)]=1 # nans and 1s
            if show_cam_usage:
                camn_used = nx.array(camn_used)

            if len(f2)!=0:
                if onefigure:
                    ax = pylab.axes([0.1, height*i+0.05,  0.8, height],sharex=ax)
                    useful_axes[cam_id] = ax
                    ax.xaxis.set_major_formatter(matplotlib.ticker.OldScalarFormatter())
                    xline3d,yline3d = ax.plot( f,  xy[0,:],'rx', # projected from 3D reconstruction
                                               f,  xy[1,:],'gx',
                                               ms=6.0,
                                               )

                    xline2d,yline2d = ax.plot( f2, x2,'ko', # real data
                                               f2, y2,'bo', mfc='None', ms=2.0 )
                    pylab.setp(xline2d,'markeredgecolor',(0.5,0.0,0.0))#'red')
                    pylab.setp(yline2d,'markeredgecolor',(0.0,0.3,0.0))#'green')

                    oriline, = ax.plot( f2, p1,'ko') # orientation data present
                    if show_cam_usage:
                        camusedline, = ax.plot( f2, 700*camn_used,'yo') # orientation data present
                        pylab.setp(camusedline,'markeredgecolor',(0.0,0.3,0.0,0.0))
                        pylab.setp(camusedline,'markeredgewidth',0)

                    pylab.ylabel( cam_id )
                    pylab.setp(ax, 'ylim',[-10,710])
                else:
                    pylab.figure() # new figure
                    lines = pylab.plot( x2, y2, 'ko',
                                        xy[0,:], xy[1,:], 'rx' ) # projected from 3D reconstruction
                    pylab.legend(lines,['original 2D data','reprojections'])
                    pylab.xlabel( 'X (pixels)' )
                    pylab.ylabel( 'Y (pixels)' )
                    pylab.title( cam_id )
                    pylab.setp(pylab.gca(), 'xlim',[-10,660])
                    pylab.setp(pylab.gca(), 'ylim',[-10,500])
        if onefigure:
            useful_cam_ids = useful_axes.keys()
            useful_cam_ids.sort()
            useful_cam_ids.reverse()
            height = 0.9/len(useful_cam_ids)
            fbycid = {}
            if hasattr(results.root,'exact_movie_info'):
                for row in results.root.exact_movie_info:
                    cam_id = row['cam_id']
                    start_frame = row['start_frame']
                    stop_frame = row['stop_frame']
                    fbycid.setdefault(cam_id,[]).append( (start_frame,stop_frame))
            for i, cam_id in enumerate(useful_cam_ids):
                ax=useful_axes[cam_id]
                ax.set_position([0.1, height*i+0.05,  0.8, height])
                if cam_id in fbycid:
                    for start_frame,stop_frame in fbycid[cam_id]:
                        ax.axvspan( start_frame, stop_frame, fc=0.5, alpha=0.5 )
    finally:
        ion()

def plot_whole_movie_3d(results, typ='best', show_err=False, max_err=10,
                        fstart=None, fstop=None):
    import flydra.reconstruct
    ioff()
    
    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best

    f,xyz,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)

    if fstart is not None:
        idx = nx.where( f >= fstart )[0]
        f = nx.take(f,idx)
        xyz = nx.take(xyz,idx)
        L = nx.take(L,idx)
        err = nx.take(err,idx)
    if fstop is not None:
        idx = nx.where( f <= fstop )[0]
        f = nx.take(f,idx)
        xyz = nx.take(xyz,idx)
        L = nx.take(L,idx)
        err = nx.take(err,idx)

    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    clf()
    yinc = .65/3
    ax_x = pylab.axes([0.1,  0.25+2*yinc,  0.8, yinc])
    ax_y = pylab.axes([0.1,  0.25+yinc,  0.8, yinc],sharex=ax_x)
    ax_z = pylab.axes([0.1,  0.25,  0.8, yinc],sharex=ax_x)
    ax_xlen = pylab.axes([0.1, 0.15,  0.8, 0.1],sharex=ax_x)
    ax_err = pylab.axes([0.1, 0.05,  0.8, 0.1],sharex=ax_x)
    
    # plot it!
    xl=ax_x.plot(f,x,'r.')
    ax_x.axhline(y=730)
    ax_x.set_ylim([0,1500])
    yl=ax_y.plot(f,y,'g.')
    ax_y.axhline(y=152.5)
    ax_y.set_ylim([0,450])
    zl=ax_z.plot(f,z,'b.')
    ax_y.set_ylabel('position (mm)')
    ax_z.set_ylim([-100,400])
##    if show_err:
##        ax.plot(f,err,'k.')
##    setp(ax,'ylim',[-10,600])

    U = flydra.reconstruct.line_direction(L)
#    ax_xlen.plot( f, U[:,0], 'r.')
    ax_xlen.plot( f, nx.arctan2(U[:,1],U[:,0]), 'r.')
    ax_xlen.set_ylabel('angle')
    ax_xlen.set_ylim([-4,4])
        
    ax_err.plot(f,err,'k.')
    ax_err.set_xlabel('frame no.')
    ax_err.xaxis.set_major_formatter(matplotlib.ticker.OldScalarFormatter())
    ax_err.set_ylabel('err\n(pixels)')
    pylab.setp(ax_err,'ylim',[0,10])
    ##ax.title(typ+' data')
    ##ax.xlabel('frame number')

    if hasattr(results.root,'exact_movie_info'):
        fbycid = {}
        for row in results.root.exact_movie_info:
            cam_id = row['cam_id']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']
            fbycid.setdefault(cam_id,[]).append( (start_frame,stop_frame))

        cam_ids = fbycid.keys()
        cam_ids.sort()
        yticks = []
        yticklabels = []
        for y, cam_id in enumerate( cam_ids ):
            pairs = fbycid[cam_id]
            for start, stop in pairs:
                plot( [start, stop], [y, y] )
            yticks.append( y )
            yticklabels.append( cam_id )
        setp( gca(), 'yticks', yticks )
        setp( gca(), 'yticklabels', yticklabels )
        if len(yticks):
            setp( gca(), 'ylim', [-1, max(yticks)+1])

    ion()

def plot_whole_range(results, start_frame, stop_frame, typ='best', show_err=False):
    ioff()
    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best

    f=[]
    x=[]
    y=[]
    z=[]
    for row in data3d:
        if start_frame<=row['frame']<=stop_frame:
            f.append(row['frame'])
            x.append(row['x'])
            y.append(row['y'])
            z.append(row['z'])
    f = nx.array(f)
    x = nx.array(x)
    y = nx.array(y)
    z = nx.array(z)
    if show_err:
        err = nx.array(data3d.cols.mean_dist)
    
    # plot it!
    ax = pylab.axes()
    ax.plot(f,x,'r.')
    ax.plot(f,y,'g.')
    ax.plot(f,z,'b.')
    if show_err:
        ax.plot(f,err,'k.')
    ##ax.title(typ+' data')
    ##ax.xlabel('frame number')
    ion()

def save_smooth_data(results,frames,Psmooth,Qsmooth,table_name='smooth_data'):
    assert len(frames)==len(Psmooth)==len(Qsmooth)
    if not hasattr(results.root,table_name):
        smooth_data = results.createTable(results.root,table_name,SmoothData,'')
    else:
        smooth_data = results.root.smooth_data
        
    for i in range(len(frames)):        
        frame = frames[i]
        P = Psmooth[i]
        Q = Qsmooth[i]

        old_nrow = None
        for row in smooth_data:
            if row['frame'] != frame:
                    continue
            print row.nrow()
            if old_nrow is not None:
                raise RuntimeError('more than row with frame number %d in smooth_data'%frame)
            old_nrow = row.nrow()
            old_nrow = int(old_nrow) # XXX pytables bug...
        
        new_row = []
        new_row_dict = {}
        for colname in smooth_data.colnames:
            if colname == 'frame': new_row.append( frame )
            elif colname == 'x': new_row.append( P[0] )
            elif colname == 'y': new_row.append( P[1] )
            elif colname == 'z': new_row.append( P[2] )
            elif colname == 'qw': new_row.append( Q.w )
            elif colname == 'qx': new_row.append( Q.x )
            elif colname == 'qy': new_row.append( Q.y )
            elif colname == 'qz': new_row.append( Q.z )
            else: raise KeyError("don't know column name '%s'"%colname)
            new_row_dict[colname] = new_row[-1]
        if old_nrow is None:
            for k,v in new_row_dict.iteritems():
                smooth_data.row[k] = v
            smooth_data.row.append()
        else:
            smooth_data[old_nrow] = new_row
    smooth_data.flush()

def save_resultant(results,frames,resultant):
    return save_timed_forces('resultant_forces',results,frames,resultant)

def save_timed_forces(table_name,results,frames,resultant):
    assert len(frames)==len(resultant)
    if not hasattr(results.root,table_name):
        resultant_forces = results.createTable(results.root,table_name,TimedVectors,'')
    else:
        resultant_forces = results.root.resultant_forces
        
    for i in range(len(frames)):        
        frame = frames[i]
        fxyz = resultant[i]

        old_nrow = None
        for row in resultant_forces:
            if row['frame'] != frame:
                    continue
            if old_nrow is not None:
                raise RuntimeError('more than row with frame number %d in smooth_data'%frame)
            old_nrow = row.nrow()
        
        new_row = []
        new_row_dict = {}
        for colname in resultant_forces.colnames:
            if colname == 'frame': new_row.append( frame )
            elif colname == 'fx': new_row.append( fxyz[0] )
            elif colname == 'fy': new_row.append( fxyz[1] )
            elif colname == 'fz': new_row.append( fxyz[2] )
            else: raise KeyError("don't know column name '%s'"%colname)
            new_row_dict[colname] = new_row[-1]
        if old_nrow is None:
            for k,v in new_row_dict.iteritems():
                resultant_forces.row[k] = v
            resultant_forces.row.append()
        else:
            resultant_forces[old_nrow] = new_row
    resultant_forces.flush()

def set_ignore_frames(results,start_frame,stop_frame):
    if not hasattr(results.root,'ignore_frames'):
        ignore_frames = results.createTable(results.root,'ignore_frames',IgnoreFrames,'')
    else:
        ignore_frames = results.root.ignore_frames

    ignore_frames.row['start_frame']=start_frame
    ignore_frames.row['stop_frame']=stop_frame
    ignore_frames.row.append()
    ignore_frames.flush()

def make_exact_movie_info2(results):
    status('making exact movie info')
    
    movie_info = results.root.movie_info
    data2d = results.root.data2d
    cam_info = results.root.cam_info

    camn2cam_id = {}
    for row in cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        camn2cam_id[camn]=cam_id

    exact_movie_info = results.createTable(results.root,'exact_movie_info',ExactMovieInfo,'')
    
    for row in movie_info:
        cam_id = row['cam_id']
        filename = row['filename']
        computer_name = cam_id.split(':')[0]
        filename = filename.replace('local',computer_name)
        print 'filename',filename
        frame_server = FlyMovieFormat.FlyMovie(filename,check_integrity=True)
        status(' for %s %s:'%(cam_id,filename))
        tmp_frame, timestamp_movie_start = frame_server.get_frame( 0 )
        tmp_frame, timestamp_movie_stop = frame_server.get_frame( -1 )
        status('  %s %s'%(repr(timestamp_movie_start),repr(timestamp_movie_stop)))
        camn_start_frame_list = [(x['camn'],x['frame']) for x in data2d
                                 if x['timestamp'] == timestamp_movie_start ]
##        camn_start_frame_list = [(x['camn'],x['frame']) for x in data2d.where(
##            data2d.cols.timestamp == timestamp_movie_start )]
        if len(camn_start_frame_list) == 0:
            status('WARNING: movie for %s %s : start data not found'%(cam_id,filename))
            #ts = nx.array(data2d.cols.timestamp)
            #print 'min(ts),timestamp_movie_start,max(ts)',min(ts),timestamp_movie_start,max(ts)
            continue
        else:
            if len(camn_start_frame_list) > 1:
                for camn, start_frame in camn_start_frame_list:
                    if camn2cam_id[camn] == cam_id:
                        break
            else:
                camn, start_frame = camn_start_frame_list[0]
            assert camn2cam_id[camn] == cam_id
        camn_stop_frame_list = [x['frame'] for x in data2d
                                if x['timestamp'] == timestamp_movie_stop ]
##        camn_stop_frame_list = [x['frame'] for x in data2d.where(
##            data2d.cols.timestamp == timestamp_movie_stop )]
        if len(camn_stop_frame_list) == 0:
            status('WARNING: movie for %s %s : stop data not found in data2d, using last data2d as stop point'%(cam_id,filename))
            camn_frame_list = [x['frame'] for x in data2d
                               if x['timestamp'] >= timestamp_movie_start ]
            stop_frame = max(camn_frame_list)
        else:
            stop_frame = camn_stop_frame_list[0]
            
        exact_movie_info.row['cam_id']=cam_id
        exact_movie_info.row['filename']=filename
        exact_movie_info.row['start_frame']=start_frame
        exact_movie_info.row['stop_frame']=stop_frame
        exact_movie_info.row['start_timestamp']=timestamp_movie_start
        exact_movie_info.row['stop_timestamp']=timestamp_movie_stop
        exact_movie_info.row.append()
        exact_movie_info.flush()

def get_movie_frame2(results, cam_timestamp_or_frame, cam, suffix=None):
    if suffix is None:
        suffix = ''
        
    if isinstance(cam_timestamp_or_frame,float):
        cam_timestamp = cam_timestamp_or_frame
    else:
        frame = cam_timestamp_or_frame
        camn, cam_timestamp = get_camn_and_timestamp(results,cam,frame)
    
    cam_info = results.root.cam_info
    movie_info = results.root.movie_info
    data2d = results.root.data2d
    data3d = results.root.data3d_best

    if isinstance(cam,int): # camn
        camn = cam
        cam_id = [x['cam_id'] for x in cam_info.where( cam_info.cols.camn == camn) ][0]
        cam_id = [x['cam_id'] for x in cam_info if x['camn'] == camn ][0]
    elif isinstance(cam,str): # cam_id
        cam_id = cam
        
    if not hasattr(results.root,'exact_movie_info'):
        make_exact_movie_info2(results)
    exact_movie_info = results.root.exact_movie_info

    # find normal (non background movie filename)
    found = False
    for row in exact_movie_info:
        if row['cam_id'] == cam_id:
            if row['start_timestamp'] < cam_timestamp < row['stop_timestamp']:
                filename = row['filename']
                found = True
                break
    if not found:
        raise ValueError('movie not found for %s'%(cam_id,))

    filename = os.path.splitext(filename)[0] + '%s.fmf'%(suffix,) # alter to be background image

    fly_movie = FlyMovieFormat.FlyMovie(filename)
    frame, movie_timestamp = fly_movie.get_frame_at_or_before_timestamp( cam_timestamp )
    return frame, movie_timestamp

def get_cam_ids(results):
    cam_info = results.root.cam_info
    cam_ids=list(sets.Set(cam_info.cols.cam_id))
    cam_ids.sort()
    return cam_ids

def recompute_3d_from_2d(results,
                         overwrite=False,
                         hypothesis_test=True, # discards camns_used
                         typ='best',
                         start_stop=None, # used for hypothesis_test
                         ):
    import flydra.reconstruct
    import reconstruct_utils as ru
    reconstructor = flydra.reconstruct.Reconstructor(results)

    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best

    data2d = results.root.data2d
    cam_info = results.root.cam_info

    camn2cam_id = {}
    for row in cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        camn2cam_id[camn]=cam_id
        
    max_n_cameras = len(reconstructor.cam_combinations[0])
    
    if not hypothesis_test:
        print len(data3d),'rows to be processed'
        count = 0
        for row in data3d:
            if count%100==0:
                print 'processing row',count
            count += 1
            camns_used = map(int,row['camns_used'].split())
            if not len(camns_used):
                continue
            nrow = row.nrow()
            frame_no = row['frame']
            d2 = {}
            for x in data2d.where( data2d.cols.frame == frame_no ):
                camn = x['camn']
                if camn in camns_used:
                    cam_id = camn2cam_id[camn]
                    d2[cam_id] = (x['x'], x['y'], x['area'], x['slope'],
                                  x['eccentricity'], x['p1'], x['p2'],
                                  x['p3'], x['p4'])

            X, line3d = reconstructor.find3d(d2.items())
            if overwrite:
                new_row = []
                for colname in data3d.colnames:
                    if colname == 'x': value = X[0]
                    elif colname == 'y': value = X[1]
                    elif colname == 'z': value = X[2]
                    else: value = row[colname]
                    if line3d is not None:
                        if   colname == 'p0': value = line3d[0]
                        elif colname == 'p1': value = line3d[1]
                        elif colname == 'p2': value = line3d[2]
                        elif colname == 'p3': value = line3d[3]
                        elif colname == 'p4': value = line3d[4]
                        elif colname == 'p5': value = line3d[5]
                    new_row.append( value )
                data3d[nrow] = new_row

    else: # do hypothesis testing
        if start_stop == 'all':
            frame_range = list(data3d.cols.frame)
        else:
            start_frame, stop_frame = start_stop
            frame_range = range(start_frame,stop_frame+1)

        n_rows = len(frame_range)
        count = 0
        for frame_no in frame_range:
            count += 1
            print 'frame_no',frame_no,'(% 5d of %d)'%(count,n_rows)
            # load all 2D data
            d2 = {}
            cam_id2camn = {} # must be recomputed each frame
            for x in data2d:
                if x['frame'] == frame_no :
                    camn = x['camn']
                    cam_id = camn2cam_id[camn]
                    cam_id2camn[cam_id] = camn
                    d2[cam_id] = (x['x'], x['y'], x['area'], x['slope'],
                                  x['eccentricity'], x['p1'], x['p2'],
                                  x['p3'], x['p4'])
            try:
                X, line3d, cam_ids_used, mean_dist = ru.find_best_3d(reconstructor,d2)
            except Exception, exc:
                print 'ERROR:',exc
                print
                continue
            camns_used = [ cam_id2camn[cam_id] for cam_id in cam_ids_used ]
            if not overwrite:
                continue
            
            # find old row
            old_nrow = None
            #for row in data3d.where( data3d.cols.frame == frame_no ):
            for row in data3d:
                if row['frame'] != frame_no:
                    continue

                if old_nrow is not None:
                    raise RuntimeError('more than row with frame number %d in data3d'%frame_no)
                old_nrow = row.nrow()

            # modify row
            if line3d is None:
                line3d = [nan]*6 # fill with nans
            cam_nos_used_str = ' '.join( map(str, camns_used) )
            new_row = []
            new_row_dict = {}
            for colname in data3d.colnames:
                if colname == 'frame': new_row.append( frame_no )
                elif colname == 'x': new_row.append( X[0] )
                elif colname == 'y': new_row.append( X[1] )
                elif colname == 'z': new_row.append( X[2] )
                elif colname == 'p0': new_row.append( line3d[0] )
                elif colname == 'p1': new_row.append( line3d[1] )
                elif colname == 'p2': new_row.append( line3d[2] )
                elif colname == 'p3': new_row.append( line3d[3] )
                elif colname == 'p4': new_row.append( line3d[4] )
                elif colname == 'p5': new_row.append( line3d[5] )
                elif colname == 'timestamp': new_row.append( 0.0 )
                elif colname == 'camns_used': new_row.append(cam_nos_used_str)
                elif colname == 'mean_dist': new_row.append(mean_dist)
                else: raise KeyError("don't know column name '%s'"%colname)
                new_row_dict[colname] = new_row[-1]
            if old_nrow is None:
                for k,v in new_row_dict.iteritems():
                    data3d.row[k] = v
                data3d.row.append()
            else:
                raise RuntimeError('This code path disabled because it de-orders the table')
                #data3d[old_nrow] = new_row
            data3d.flush()

def get_reconstructor(results):
    import flydra.reconstruct
    return flydra.reconstruct.Reconstructor(results)

def plot_all_images(results,
                    frame_no,
                    show_raw_image=True,
                    zoomed=True,
                    typ='best',
                    PLOT_RED=True,  # raw 3d data
                    PLOT_BLUE=True, # smoothed 3d data
                    ##recompute_3d=False,
                    frame_server_dict=None,
                    do_undistort=True,
                    fixed_im_centers=None,
                    fixed_im_lims=None,
                    plot_orientation=True,
                    plot_true_3d_line=False, # show real line3d info (don't adjust to intersect 3d point)
                    plot_3d_unit_vector=True,
                    origin='upper', # upper is right-side up, lower works better in mpl (no y conversions)
#                    origin='lower',
                    display_labels=True,
                    display_titles=True,
                    start_frame_offset=188191, # used to calculate time display
                    max_err=None,
                    plot_red_ori_fixed=False,                    
                    colormap='jet'):

    #if origin == 'upper' and zoomed==True: raise NotImplementedError('')
    
    if fixed_im_centers is None:
        fixed_im_centers = {}
    if fixed_im_lims is None:
        fixed_im_lims = {}
    ioff()
    import flydra.reconstruct
    reconstructor = flydra.reconstruct.Reconstructor(results)

    if typ == 'fast':
        data3d = results.root.data3d_fast
    elif typ == 'best':
        data3d = results.root.data3d_best

    data2d = results.root.data2d
    cam_info = results.root.cam_info

    if colormap == 'jet':
        cmap = cm.jet
    elif colormap.startswith('gray'):
        cmap = cm.gray
    else:
        raise ValueError("unknown colormap '%s'"%colormap)

    camn2cam_id = {}
    for row in cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        camn2cam_id[camn]=cam_id
    
    # find total number of cameras plugged in:
    cam_ids=list(sets.Set(cam_info.cols.cam_id))
    cam_ids.sort()

    tmp = [((x['x'],x['y'],x['z']),
            (x['p0'],x['p1'],x['p2'],x['p3'],x['p4'],x['p5']),
            x['camns_used'], x['mean_dist']
            ) for x in data3d.where( data3d.cols.frame == frame_no) ]
    if len(tmp) == 0:
        X, line3d = None, None
        camns_used = ()
    else:
        assert len(tmp)==1
        X, line3d, camns_used, err = tmp[0]
        camns_used = map(int,camns_used.split())
        if max_err is not None:
            if err > max_err:
                X, line3d = None, None
                camns_used = ()
        
    clf()
    figtext( 0.5, 0.99, '% 5.2f sec'%( (frame_no-start_frame_offset)/100.0,),
             horizontalalignment='center',
             verticalalignment='top',
             )
    for subplot_number,cam_id in enumerate(cam_ids):
#        print cam_id
        width, height = reconstructor.get_resolution(cam_id)
        
        i = cam_ids.index(cam_id)
        ax=dougs_subplot(subplot_number)
        setp(ax,'frame_on',display_labels)

        have_2d_data = False
        nan_in_2d_data = False
##        for row in data2d.where( data2d.cols.frame == frame_no ):        
        for row in data2d:
            if row['frame'] != frame_no:
                # XXX ARGH!!! some weird bug in my code or pytables??
                # it means I can't do "in kernel", e.g.:
                #        for row in data2d.where( data2d.cols.frame == frame_no ):
                continue
            camn = row['camn']
            if camn2cam_id[camn] == cam_id:
                have_2d_data = True
                x=row['x']
                y=row['y']
                slope=row['slope']
                eccentricity=row['eccentricity']
                remote_timestamp = row['timestamp']
                if len(getnan([x])[0]):
                    nan_in_2d_data = True
                break
        if not have_2d_data:
            camn=None
            x=None
            y=None
            slope=None
            eccentricity=None
            remote_timestamp = None
            nan_in_2d_data = True
            
        title_str = cam_id

        have_limit_data = False
        if show_raw_image:
            have_raw_image = False
            if have_2d_data:
                try:
                    # XXXXX
                    im, movie_timestamp = get_movie_frame2(results, remote_timestamp, cam_id)#, frame_server_dict=frame_server_dict)
                    have_raw_image = True
                except ValueError,exc:
                    print 'WARNING: skipping frame for %s because ValueError: %s'%(cam_id,str(exc))
                except KeyError,exc:
                    print 'WARNING: skipping frame for %s because KeyError: %s'%(cam_id,str(exc))
            if have_2d_data and have_raw_image:
                if remote_timestamp != movie_timestamp:
                    print 'Whoa! timestamps are not equal!',cam_id
                    print ' XXX may be able to fix display, but not displaying wrong image for now'
            if have_raw_image:
#                print cam_id,'have_raw_image True'
                if do_undistort:
                    intrin = reconstructor.get_intrinsic_linear(cam_id)
                    k = reconstructor.get_intrinsic_nonlinear(cam_id)
                    f = intrin[0,0], intrin[1,1] # focal length
                    c = intrin[0,2], intrin[1,2] # camera center
                    im = undistort.rect(im, f=f, c=c, k=k)
                    im = im.astype(nx.UInt8)
                if (have_2d_data and not nan_in_2d_data and zoomed and x>=0
                    or fixed_im_lims.has_key(cam_id)):
                    w = 13
                    h = 20
                    xcenter, ycenter = fixed_im_centers.get(
                        cam_id, (x,y))
                    xmin = int(xcenter-w)
                    xmax = int(xcenter+w)
                    ymin = int(ycenter-h)
                    ymax = int(ycenter+h)

                    # workaround ipython -pylab mode:
                    max = sys.modules['__builtin__'].max
                    min = sys.modules['__builtin__'].min
                    
                    xmin = max(0,xmin)
                    xmax = min(xmax,width-1)
                    ymin = max(0,ymin)
                    ymax = min(ymax,height-1)

                    (xmin, xmax), (ymin, ymax) = fixed_im_lims.get(
                        cam_id, ((xmin,xmax),(ymin,ymax)) )
                    
                    if origin == 'upper':
                        show_ymin = height-ymax
                        show_ymax = height-ymin
                    else:
                        show_ymin = ymin
                        show_ymax = ymax
                    
                    have_limit_data = True

                    if origin == 'upper':
                        extent = (xmin,xmax,height-ymin,height-ymax)
                    else:
                        extent = (xmin,xmax,ymin,ymax)
                    im = im.copy() # make contiguous
                    im_small = im[ymin:ymax,xmin:xmax]
                    if origin == 'upper':
                        im_small = im_small[::-1] # flip-upside down
                    im_small = im_small.copy()
                    ax.imshow(im_small,
                              origin=origin,
                              interpolation='nearest',
                              cmap=cmap,
                              extent = extent,
                              )
                    
                else:
                    #print 'XXX code path 1',cam_id
                    xmin=0
                    xmax=width-1
                    ymin=0
                    ymax=height-1

##                    (xmin, xmax), (ymin, ymax) = fixed_im_lims.get(
##                        cam_id, ((xmin,xmax),(ymin,ymax)) )
                    
                    show_ymin = ymin
                    show_ymax = ymax
                    
                    if origin == 'upper':
                        extent = (xmin,xmax,height-ymax,height-ymin)
                    else:
                        extent = (xmin,xmax,ymin,ymax)

                    ax.imshow(im,
                              origin=origin,
                              interpolation='nearest',
                              extent=extent,
                              cmap=cmap)
            else:
                #print 'XXX code path 2',cam_id
                    
                xmin=0
                xmax=width-1
                ymin=0
                ymax=height-1
                
                (xmin, xmax), (ymin, ymax) = fixed_im_lims.get(
                    cam_id, ((xmin,xmax),(ymin,ymax)) )
                
                show_ymin = ymin
                show_ymax = ymax

        if have_2d_data and not nan_in_2d_data and camn is not None:
                
            # raw 2D
            try:
                if origin == 'upper':
                    lines=ax.plot([x],[height-y],'o')
                else:
                    lines=ax.plot([x],[y],'o')
            except:
                print 'x, y',x, y
                sys.stdout.flush()
                raise
            
            if show_raw_image:
                green = (0,1,0)
                setp(lines,'markerfacecolor',None)
                if camn in camns_used:
                    setp(lines,'markeredgecolor',green)
                elif camn not in camns_used:
                    setp(lines,'markeredgecolor',(0, 0.2, 0))
                    #print 'setting alpha in',cam_id
                    #setp(lines,'alpha',0.2)
                setp(lines,'markeredgewidth',2.0)

            #if not len(numarray.ieeespecial.getnan(slope)[0]):
            if eccentricity > flydra.reconstruct.MINIMUM_ECCENTRICITY:
                #title_str = cam_id + ' %.1f'%eccentricity
                
                #eccentricity = min(eccentricity,100.0) # bound it

                # ax+by+c=0
                a=slope
                b=-1
                c=y-a*x

                x1=xmin
                y1=-(c+a*x1)/b
                if y1 < ymin:
                    y1 = ymin
                    x1 = -(c+b*y1)/a
                elif y1 > ymax:
                    y1 = ymax
                    x1 = -(c+b*y1)/a
                
                x2=xmax
                y2=-(c+a*x2)/b
                if y2 < ymin:
                    y2 = ymin
                    x2 = -(c+b*y2)/a
                elif y2 > ymax:
                    y2 = ymax
                    x2 = -(c+b*y2)/a                

                if plot_orientation:
                    if origin == 'upper':
                        lines=ax.plot([x1,x2],[height-y1,height-y2],':',linewidth=1.5)
                    else:
                        lines=ax.plot([x1,x2],[y1,y2],':',linewidth=1.5)
                    if show_raw_image:
                        green = (0,1,0)
                        if camn in camns_used:
                            setp(lines,'color',green)
                        elif camn not in camns_used:
                            setp(lines,'color',(0, 0.2, 0))
                        #setp(lines,'color',green)
                        #setp(lines[0],'linewidth',0.8)

        if X is not None:
            if line3d is None:
                x,y=reconstructor.find2d(cam_id,X)
                l3=None
            else:
                if plot_true_3d_line:
                    x,l3=reconstructor.find2d(cam_id,X,line3d)
                else:
                    U = flydra.reconstruct.line_direction(line3d)
                    if plot_3d_unit_vector:
                        x=reconstructor.find2d(cam_id,X)
                        unit_x1, unit_y1=reconstructor.find2d(cam_id,X-5*U)
                        unit_x2, unit_y2=reconstructor.find2d(cam_id,X+5*U)
                    else:
                        line3d_fake = flydra.reconstruct.pluecker_from_verts(X,X+U)
                        x,l3=reconstructor.find2d(cam_id,X,line3d_fake)
                    x,y=x
                
            #near = 10
            if PLOT_RED:
                if origin=='upper':
                    lines=ax.plot([x],[height-y],'o')
                else:
                    lines=ax.plot([x],[y],'o')
                setp(lines,'markerfacecolor',(1,0,0))
                setp(lines,'markeredgewidth',0.0)
                setp(lines,'markersize',4.0)
                
            if PLOT_RED and line3d is not None:
                if plot_orientation:
                    if plot_3d_unit_vector:
                        if origin == 'upper':
                            if plot_red_ori_fixed:
                                lines=ax.plot([x,unit_x1],[height-y,height-unit_y1],'r-',linewidth=1.5)
                            else:
                                lines=ax.plot([unit_x1,unit_x2],[height-unit_y1,height-unit_y2],'r-',linewidth=1.5)
                        else:
                            if plot_red_ori_fixed:
                                lines=ax.plot([x,unit_x1],[y,unit_y1],'r-',linewidth=1.5)
                            else:
                                lines=ax.plot([unit_x1,unit_x2],[unit_y1,unit_y2],'r-',linewidth=1.5)
                    else:
                        a,b,c=l3
                        # ax+by+c=0

                        # y = -(c+ax)/b
                        # x = -(c+by)/a


                        x1=xmin
                        y1=-(c+a*x1)/b
                        if y1 < ymin:
                            y1 = ymin
                            x1 = -(c+b*y1)/a
                        elif y1 > ymax:
                            y1 = ymax
                            x1 = -(c+b*y1)/a

                        x2=xmax
                        y2=-(c+a*x2)/b
                        if y2 < ymin:
                            y2 = ymin
                            x2 = -(c+b*y2)/a
                        elif y2 > ymax:
                            y2 = ymax
                            x2 = -(c+b*y2)/a

                        if origin == 'upper':
                            lines=ax.plot([x1,x2],[height-y1,height-y2],'r--',linewidth=1.5)
                        else:
                            lines=ax.plot([x1,x2],[y1,y2],'r--',linewidth=1.5)
        if PLOT_BLUE:
            smooth_data = results.root.smooth_data
            have_smooth_data = False
            for row in smooth_data:
                if row['frame'] == frame_no:
                    Psmooth = nx.array( (row['x'], row['y'], row['z']) )
                    Qsmooth = cgtypes.quat( row['qw'], row['qx'], row['qy'], row['qz'] )
                    have_smooth_data = True
                    break
            if not have_smooth_data:
                # make sure we don't use old data
                Psmooth = None
                Qsmooth = None
            else:
                #x,l3=reconstructor.find2d(cam_id,Psmooth,line3d)
                U=nx.array(PQmath.quat_to_orient(Qsmooth))
                x,y            = reconstructor.find2d(cam_id,Psmooth)
                unit_x, unit_y = reconstructor.find2d(cam_id,(Psmooth-5.0*U))

                if origin=='upper':
                    lines=ax.plot([x],[height-y],'o')
                else:
                    lines=ax.plot([x],[y],'o')
                setp(lines,'markerfacecolor',(0,0,1))

                if origin == 'upper':
                    lines=ax.plot([x,unit_x],[height-y,height-unit_y],'b-',linewidth=1.5)
                else:
                    lines=ax.plot([x,unit_x],[y,unit_y],'b-',linewidth=1.5)
                
        labels=ax.get_xticklabels()
        setp(labels, rotation=90)
        if display_titles:
            title(title_str)
        if have_limit_data:
            ax.xaxis.set_major_locator( LinearLocator(numticks=5) )
            ax.yaxis.set_major_locator( LinearLocator(numticks=5) )
            setp(ax,'xlim',[xmin, xmax])
            setp(ax,'ylim',[show_ymin, show_ymax])
        elif fixed_im_lims.has_key(cam_id):
            (xmin, xmax), (ymin, ymax) = fixed_im_lims[cam_id]
            ax.xaxis.set_major_locator( LinearLocator(numticks=5) )
            ax.yaxis.set_major_locator( LinearLocator(numticks=5) )
            setp(ax,'xlim',[xmin, xmax])
            setp(ax,'ylim',[show_ymin, show_ymax])
        else:
            margin_pixels = 20
            setp(ax,'xlim',[-margin_pixels, width+margin_pixels])
            setp(ax,'ylim',[-margin_pixels, height+margin_pixels])
        if not display_labels:
            setp(ax,'xticks',[])
            setp(ax,'yticks',[])
    ion()

def test():
    import flydra.reconstruct
    frames = get_frames_with_3d(results)
    reconstructor = flydra.reconstruct.Reconstructor(results)
    for frame in frames:
        try:
            redo_3d_calc(results,frame,reconstructor=reconstructor,
                         verify=True,overwrite=False)
        except Exception, x:
            status('ERROR (frame %d): %s'%(frame,str(x)))

def get_results(filename,mode='r+'):
    h5file = PT.openFile(filename,mode=mode)
    if hasattr(h5file.root,'data3d_best'):
        frame_col = h5file.root.data3d_best.cols.frame
        if frame_col.index is None:
            print 'creating index on data3d_best...'
            frame_col.createIndex()
            print 'done'
        
    if hasattr(h5file.root,'data2d'):
        frame_col = h5file.root.data2d.cols.frame
        if frame_col.index is None:
            print 'creating index on data2d...'
            frame_col.createIndex()
            print 'done'
    return h5file

def plot_simple_phase_plots(results,form='xy',max_err=10,typ='best',ori_180_ambig=True):
    from matplotlib.collections import LineCollection
    f,xyz,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)
    import flydra.reconstruct
    U = flydra.reconstruct.line_direction(L) # unit vector
    if form == 'xy':
        hidx = 0
        hname = 'X (mm)'
        vidx = 1
        vname = 'Y (mm)'
    elif form == 'xz':
        hidx = 0
        hname = 'X (mm)'
        vidx = 2
        vname = 'Z (mm)'
    plot(xyz[:,hidx],xyz[:,vidx],'o',mec=(0,0,0),mfc=None,ms=2.0)
    segments = []
    for i in range(len(U)):
        pi = xyz[i]
        Pqi = U[i]
        if len(getnan(pi)[0]) or len(getnan(Pqi)[0]):
            continue
        if ori_180_ambig:
            segment = ( (pi[hidx]+Pqi[hidx]*1,   # x1
                         pi[vidx]+Pqi[vidx]*1),   # y1
                        (pi[hidx]-Pqi[hidx]*1,   # x2
                         pi[vidx]-Pqi[vidx]*1) ) # y2
        else:
            segment = ( (pi[hidx],  # x1
                         pi[vidx]), # y1
                        (pi[hidx]-Pqi[hidx]*2,   # x2
                         pi[vidx]-Pqi[vidx]*2) ) # y2
        #print segment
        segments.append( segment )
    collection = LineCollection(segments)#,colors=[ (0,0,1) ] *len(segments))
    gca().add_collection(collection)
    xlabel(hname)
    ylabel(vname)

def save_movie(results):
##    fixed_im_centers = {'cam1:0':(260,304),
##                    'cam2:0':(402,226),
##                    'cam3:0':(236,435),
##                    'cam4:0':(261,432),
##                    'cam5:0':(196,370)}
    full_frame = ((0,655),(0,490))
    fixed_im_lims = {
        'cam1:0':((70,  170), (200, 270)),
        'cam2:0':((350, 430), ( 80, 220)),
        'cam3:0':((350, 450), (  0, 160)),
        'cam4:0':((290, 400), ( 20, 180)),
        'cam5:0':((355, 395), (130, 200)),
        }

    cam_ids = get_cam_ids(results,)

    start_frame = 1231000
    stop_frame = 1231800
        
    for frame in xrange(start_frame+240, 1231000+300, 1):
        clf()
        try:
            fname = 'zoomed/zoomed_%04d.png'%frame
            #fname = 'full_frame_%04d.png'%frame
            print ' plotting',fname,'...',
            sys.stdout.flush()
            plot_all_images(results, frame,
                            #frame_server_dict=frame_server_dict,
                            #fixed_im_centers=fixed_im_centers,
                            #fixed_im_lims=fixed_im_lims,
                            colormap='grayscale',
                            #zoomed=True,
                            
                            plot_orientation=True,
#                            plot_orientation=False,
                            
                            origin='lower',
                            display_labels=False,
                            display_titles=True,
                            start_frame_offset=start_frame,
                            PLOT_RED=True,
                            PLOT_BLUE=False,
                            max_err=10,
                            plot_red_ori_fixed=False,                    
                            )
            
            print ' saving...',
            sys.stdout.flush()
            savefig(fname)
            print 'done'
        except Exception, x:
            #print x, str(x)
            raise

def plot_camera_view(results,camn):
    ioff()
    try:
        start_frame = 68942+330
        stop_frame = 68942+360
        
        f1 = start_frame
        f2 = stop_frame

        for row in results.root.cam_info:
            if camn == row['camn']:
                cam_id = row['cam_id']

        f = []
        x = []
        y = []
        cam_timestamps = []
        for row in results.root.data2d:
            if row['camn'] != camn:
                continue
            if f1<=row['frame']<=f2:
                if len( getnan(row['x'])[0] ) == 0:
                    f.append( row['frame'] )
                    x.append( row['x'] )
                    y.append( row['y'] )
                    cam_timestamps.append( row['timestamp'] )
        plot(x,y,'o-',mfc=None,mec='k',ms=2.0)
        for i,frame in enumerate(f):
            t = (frame-start_frame) / 100.0
            #if (t%0.1) < 1e-5 or (t%0.1)>(0.1-1e-5):
            if 1:
                text( x[i], y[i], str(t) )
        title(cam_id)
    finally:
        ion()
    return f, cam_timestamps
    
def get_data_array(results):
##    import flydra.reconstruct
##    save_ascii_matrix
    
    data3d = results.root.data3d_best

    M = []
    for row in data3d.where( 132700 <= data3d.cols.frame <= 132800 ):
        M.append( (row['frame'], row['x'], row['y'], row['z'] ) )
    M = nx.array(M)
    return M

def get_start_stop_times( results ):
    data3d = results.root.data3d_best
    
    # XXX Assume all rows in table are in chronological
    # order. (Messing with tables during analysis could have screwed
    # this up.)

    try:
        results = data3d.cols.timestamp[0], data3d.cols.timestamp[-1]
    except IndexError:
        results = None
        
    return results

##    all_times = data3d.cols.timestamp
##    timin = nx.argmin(all_times)
##    timax = nx.argmax(all_times)
##    return all_times[timin], all_times[timax]
    
def get_timestamp( results, frame, cam):
    camn2cam_id = {}
    for row in results.root.cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        camn2cam_id[camn]=cam_id

    found = False
    if isinstance(cam,int):
        camn = cam
        for row in results.root.data2d:
            if row['frame'] == frame and row['camn'] == camn:
                timestamp = row['timestamp']
                found = True

    else:
        cam_id = cam
        for row in results.root.data2d:
            if row['frame'] == frame:
                camn = row['camn']
                if camn2cam_id[camn] == cam_id:
                    if found:
                        print 'WARNING: multiple frames found with same'
                        print 'timestamp and cam_id. (Use camn instead.)'
                    timestamp = row['timestamp']
                    found = True
    if found:
        return timestamp
    else:
        return None

class RT_Analyzer_State:
    def __init__(self, results, camn, diff_threshold, clear_threshold, start_frame):
        cam_info = results.root.cam_info
        cam_id = [x['cam_id'] for x in cam_info if x['camn'] == camn ][0]

        frame, timestamp, self.rt_state = get_frame_ts_and_realtime_analyzer_state( results,
                                                                                    frame = start_frame,
                                                                                    camn = camn,
                                                                                    diff_threshold=diff_threshold,
                                                                                    clear_threshold=clear_threshold,
                                                                         )
        self.cur_frame_no = start_frame
        self.cur_image = frame

        fg_frame_server = get_server(cam_id)
        bg_frame_server = get_server(cam_id,port=9899) # port 9889 for bg images
        self.frame_server_dict_fg = { cam_id:fg_frame_server }
        self.frame_server_dict_bg = { cam_id:bg_frame_server }
        

def get_frame_ts_and_realtime_analyzer_state( results,
                                              frame = 6804,
                                              camn = 15,
                                              diff_threshold=None,
                                              clear_threshold=None,
                                              ):
    timestamp = None
    for row in results.root.data2d:
        if row['frame'] == frame and row['camn'] == camn:
            timestamp = row['timestamp']
    if timestamp is None:
        return None,None,None
    print 'timestamp',timestamp
            
    cam_info = results.root.cam_info      
    cam_id = [x['cam_id'] for x in cam_info if x['camn'] == camn ][0]

    import realtime_image_analysis4
    import flydra.reconstruct
    import FastImage
    
    frame,timestamp2=get_movie_frame2(results, timestamp, camn, suffix='')
    assert timestamp2-timestamp < 1e-15
    bg_frame,bg_timestamp=get_movie_frame2(results, timestamp, camn, suffix='_bg')
    bg_frame_fi = FastImage.asfastimage(bg_frame)
    std_frame,std_timestamp=get_movie_frame2(results, timestamp, camn, suffix='_std')
    std_frame_fi = FastImage.asfastimage(std_frame)
    
    if 0:
        diff = frame.astype(numarray.Int32) - bg_frame.astype(numarray.Int32)
        imshow(diff)
        colorbar()
    
    reconstructor = flydra.reconstruct.Reconstructor(results)
    
    ALPHA = 0.1
    rt = realtime_image_analysis4.RealtimeAnalyzer(frame.shape[1],frame.shape[0])
    
    mean_im = rt.get_image_view('mean')
    bg_frame_fi.get_8u_copy_put(mean_im,bg_frame_fi.size)

    cmp_im = rt.get_image_view('cmp')
    std_frame_fi.get_8u_copy_put(cmp_im,std_frame_fi.size)
    
    rt.set_reconstruct_helper( reconstructor.get_recontruct_helper_dict()[cam_id] )
    rt.pmat = reconstructor.get_pmat(cam_id)
    if clear_threshold is None:
        clear_threshold = 0.9 # XXX should save to fmf file??
        print 'WARNING: set clear_threshold to',clear_threshold
    if diff_threshold is None:
        diff_threshold = 15 # XXX should save to fmf file??
        print 'WARNING: set diff_threshold to',diff_threshold
    rt.clear_threshold = clear_threshold
    rt.diff_threshold = diff_threshold
    return frame, timestamp, rt

def show_working_image(results,cam,fno,
                       diff_threshold=15.0,
                       clear_threshold=0.2):
    if isinstance(cam,int):
        camn = cam
    else:
        orig_cam_id = cam
        
        cam_id2camns = {}
        for row in results.root.cam_info:
            add_cam_id, add_camn = row['cam_id'], row['camn']
            cam_id2camns.setdefault(add_cam_id,[]).append(add_camn)

        found = False
        for row in results.root.data2d.where(results.root.data2d.cols.frame==fno):
            test_camn = row['camn']
            if test_camn in cam_id2camns[orig_cam_id]:
                camn = test_camn
                found = True
                break

        if not found:
            raise ValueError("could not find data for cam")
                    
    use_roi2 = True
    frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                              fno, camn,
                                                              diff_threshold,
                                                              clear_threshold)
    points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
    wi = rt.get_working_image()
    imshow(wi,interpolation='nearest',origin='lower')
    colorbar()
    return points[0]

def recompute_2d_data(results,camn,start_frame,stop_frame,
                      diff_threshold=15.0,
                      clear_threshold=0.2):
    data2d = results.root.data2d
    use_roi2 = True
    for fno in xrange(start_frame,stop_frame+1):
        
        # get original data
        nrow = None
        orig_x = None
        orig_y = None
        for row in data2d:
            if row['frame']==fno and row['camn']==camn:
                nrow = row.nrow()
                orig_x, orig_y = row['x'], row['y']

        try:
            frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                                      fno, camn,
                                                                      diff_threshold,
                                                                      clear_threshold)
        except KeyError, exc:
            print 'WARNING: KeyError for frame %d, skipping'%(fno,)
            continue
        points, found, orientation = rt.do_work(frame,ts,fno,use_roi2)
        del frame
        del rt
        pt = points[0]
##        if nrow is not None:
##            x,y = pt[0:2]
##            dist = math.sqrt((orig_x-x)**2 + (orig_y-y)**2)
##            print 'new 2d point shifted %.1f pixels'

        # get data ready for pytables
        new_row = []
        new_row_dict = {}
        for colname in data2d.colnames:
            value = None
            if colname == 'camn': value = camn
            elif colname == 'frame': value = fno
            elif colname == 'timestamp': value = ts
            elif colname == 'x': value = pt[0]
            elif colname == 'y': value = pt[1]
            elif colname == 'area': value = pt[2]
            elif colname == 'slope': value = pt[3]
            elif colname == 'eccentricity': value = pt[4]
            elif colname == 'p1': value = pt[5]
            elif colname == 'p2': value = pt[6]
            elif colname == 'p3': value = pt[7]
            elif colname == 'p4': value = pt[8]
            assert value is not None
            new_row.append( value )
            new_row_dict[colname] = value
        if nrow is None:
            for k,v in new_row_dict.iteritems():
                data2d.row[k] = v
            data2d.row.append()
            print 'row appended to data2d -- no longer in order'
        else:
            data2d[nrow] = new_row
            print 'replaced row (frame %d) data with x,y='%(fno,),pt[:2]
        data2d.flush()
            
def save_ecc(results):
    import flydra.reconstruct, flydra.undistort
    ioff()
    try:
        rcn = flydra.reconstruct.Reconstructor(results)
        cam_id = 'cam4:0'
        camn = 19
        use_roi2 = True
        #for roi2_radius in [5,10,15]:
        for roi2_radius in [5]:
            for fno in range(7440+549,7440+550):
    #        for fno in range(10000+600,10000+640):
                print fno
                clf()
                frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                                          fno, camn,
                                                                          9.0, 0.2 )
                rt.roi2_radius = roi2_radius
                points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
                bright_point = rt.get_last_bright_point()
                wi = rt.get_working_image()
                pt = points[0]
                x0_abs, y0_abs, area, slope, eccentricity, p1, p2, p3, p4 = pt

                x = x0_abs
                y = y0_abs

                xmin = bright_point[0]-roi2_radius
                xmax = bright_point[0]+roi2_radius
                ymin = bright_point[1]-roi2_radius
                ymax = bright_point[1]+roi2_radius

                a=slope
                b=-1
                c=y-a*x

                x1=xmin
                y1=-(c+a*x1)/b
                if y1 < ymin:
                    y1 = ymin
                    x1 = -(c+b*y1)/a
                elif y1 > ymax:
                    y1 = ymax
                    x1 = -(c+b*y1)/a

                x2=xmax
                y2=-(c+a*x2)/b
                if y2 < ymin:
                    y2 = ymin
                    x2 = -(c+b*y2)/a
                elif y2 > ymax:
                    y2 = ymax
                    x2 = -(c+b*y2)/a 

                title('%d %f'%(fno,eccentricity))
                imshow(flydra.undistort.undistort(rcn,wi,cam_id),
                       interpolation='nearest',origin='lower')
                plot([x0_abs],[y0_abs],'o',mfc=None,mec='white',mew=2)
                plot([x1,x2],[y1,y2],'w-',lw=2)
                setp(gca(),'xlim',[xmin,xmax])
                setp(gca(),'ylim',[ymin,ymax])
                savefig('roi%02d_ecc%d_%s.png'%(roi2_radius,fno,cam_id))
    finally:
        ion()
        
def wi_test2(results):
    import flydra.reconstruct, flydra.undistort
    rcn = flydra.reconstruct.Reconstructor(results)
    cam_id = 'cam4:0'
    camn = 19
    use_roi2 = True
    roi2_radius = 15
    fno = 7440+549
    res = {}
    vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for clear_threshold in vals:
        frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                                  fno, camn,
                                                                  9.0, clear_threshold )
        rt.roi2_radius = roi2_radius
        points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
        
        area = points[0][2]
        print clear_threshold, area

        bright_point = rt.get_last_bright_point()
        wi = rt.get_working_image()

        xmin = bright_point[0]-roi2_radius
        xmax = bright_point[0]+roi2_radius
        ymin = bright_point[1]-roi2_radius
        ymax = bright_point[1]+roi2_radius

        wi2 = wi[ymin:ymax,xmin:xmax].astype(nx.Float)
        
        res[clear_threshold] = wi2
    return res
    
def get_wi(results):
    import numarray.convolve as conv_mod
    import flydra.reconstruct, flydra.undistort
    rcn = flydra.reconstruct.Reconstructor(results)
    cam_id = 'cam4:0'
    camn = 19
    use_roi2 = True
    roi2_radius = 25
    fno = 7440+549
    frame, ts, rt = get_frame_ts_and_realtime_analyzer_state( results,
                                                              fno, camn,
                                                              9.0, 0.2 )
    rt.roi2_radius = roi2_radius
    points, found, orientation = rt.do_work(frame,0,fno,use_roi2)
    bright_point = rt.get_last_bright_point()
    wi = rt.get_working_image()
    
    xmin = bright_point[0]-roi2_radius
    xmax = bright_point[0]+roi2_radius
    ymin = bright_point[1]-roi2_radius
    ymax = bright_point[1]+roi2_radius

    wi2 = wi[ymin:ymax,xmin:xmax].astype(nx.Float)
    print 'points',points
    print 'bright_point',bright_point

    kernel = gaussian( nx.arange(5)-2, 2.0 )
    kernel = kernel/ nx.sum( kernel ) # normalize

    wi3 = []
    for row in range(wi2.shape[0]):
        res = conv_mod.convolve(wi2[row,:], kernel, mode=conv_mod.VALID )
        wi3.append( res )
    wi3 = nx.array(wi3)

    wi4 = []
    for col in range(wi3.shape[1]):
        res = conv_mod.convolve(wi3[:,col], kernel, mode=conv_mod.VALID )
        wi4.append( res )
    wi4 = nx.array(wi4)
    wi4.transpose()
    
    return wi, wi2, wi4

def gaussian(x,sigma):
    return nx.exp(-x**2/sigma**2)

def cam_usage(results,typ='best'):
    
    start_frame = 68942+330
    stop_frame = 68942+360
    
    if typ=='best':
        data3d = results.root.data3d_best
    elif typ=='fastest':
        data3d = results.root.data3d_fastest
        
    data2d = results.root.data2d
    cam_info = results.root.cam_info
    
    camn2cam_id = {}
    
    for row in cam_info:
        cam_id, camn = row['cam_id'], row['camn']
        camn2cam_id[camn]=cam_id
        
    for frame in range(start_frame, stop_frame+1):
        tmp_res = [ (row['camns_used'],row['mean_dist']) for row in data3d.where(data3d.cols.frame == frame) ]
        if len(tmp_res) == 0:
            continue
        assert len(tmp_res) == 1
        camns_used = map(int,tmp_res[0][0].split(' '))
        err = tmp_res[0][1]
        
#        print 'camns_used',camns_used
        used = [int(camn2cam_id[camn][3]) for camn in camns_used]
#        print 'used',used

        camns_found = []
        for row in data2d.where(data2d.cols.frame == frame):
            if not len(getnan([row['x']])[0]):
                camn = row['camn']
                camns_found.append( row['camn'] )
                cam_id = camn2cam_id[camn]
                num = int(cam_id[3])
                
        found_but_not_used = []
        for camn in camns_found:
            if camn not in camns_used:
                cam_id = camn2cam_id[camn]
                num = int(cam_id[3])
                found_but_not_used.append(num)
#        print 'found_but_not_used',found_but_not_used
        if 1:
        #if len(found_but_not_used):
            print 'frame %d:'%(frame,),
            
            found_but_not_used.sort()
            for i in range( 6 ):
                if i in found_but_not_used:
                    print 'X',
                elif i in used:
                    print '.',
                else:
                    print ' ',
                print '  ',
            print '% 3.1f'%err


def calculate_3d_point(results, frame_server_dict=None):
    by_cam_id = {}
    for row in results.root.exact_movie_info:
        print row
        cam_id = row['cam_id']
        if by_cam_id.has_key( cam_id ):
            continue # already did this camera

        if frame_server_dict is None:
            frame_server = get_server(cam_id)
        else:
            frame_server = frame_server_dict[cam_id]

        frame_server.load( row['filename'] )
        frame, timestamp = frame_server.get_frame(0)
        by_cam_id[cam_id] = frame

    clf()
    i = 0
    for cam_id, frame in by_cam_id.iteritems():
        i += 1
        subplot(2,3,i)

        cla()
        imshow(frame)
        title(cam_id)

    return

def plot_3d_point(results, X=None, frame_server_dict=None):
    if X is not None:
        import flydra.reconstruct
        reconstructor = flydra.reconstruct.Reconstructor(results)
        
    by_cam_id = {}
    for row in results.root.exact_movie_info:
        print row
        cam_id = row['cam_id']
        if by_cam_id.has_key( cam_id ):
            continue # already did this camera

        if frame_server_dict is None:
            frame_server = get_server(cam_id)
        else:
            frame_server = frame_server_dict[cam_id]

        frame_server.load( row['filename'] )
        frame, timestamp = frame_server.get_frame(0)
        by_cam_id[cam_id] = frame

    clf()
    i = 0
    for cam_id, frame in by_cam_id.iteritems():
        i += 1
        subplot(2,3,i)

        cla()
        undistorted = flydra.undistort.undistort(reconstructor,frame,cam_id)
        imshow(undistorted)
        title(cam_id)

        if X is not None:
            xy=reconstructor.find2d(cam_id,X)
            print xy
            x,y=xy[:2]
            plot( [x], [y], 'wo')
        #ion()
    return

if 0:
        class MyPickClass:
            def __init__(self, d, cam_id):
                self.d = d
                self.cam_id = cam_id
            def pick(self,event):
                if event.key=='p' and event.inaxes is not None:
                    ax = event.inaxes
                    print self.cam_id, (event.x, event.y)
                    self.d[self.cam_id] = (event.x, event.y)

        picker = MyPickClass(by_cam_id,cam_id)
        mpl_id = pylab.connect('key_press_event',picker.pick)
                         
        try:
            print 'connected MPL event',mpl_id
            pylab.show()
            print 'Press "p" to display cursor coordinates, press enter for next frame',mpl_id
            raw_input()
        finally:
            pylab.disconnect( mpl_id )
            print 'disconnected MPL event',mpl_id
        print by_cam_id

def get_usable_startstop(results,min_len=100,max_break=5,max_err=10,typ='best'):
    f,xyz,L,err = get_f_xyz_L_err(results,max_err=max_err,typ=typ)
    del xyz
    del L
    
    sort_order = nx.argsort( f )
    f = f[sort_order]
    err = err[sort_order]
    
    good_frames = nx.take(f,nx.where( err < 10.0 ))
    good_frames = good_frames[0] # make 1D array

    f_diff = good_frames[1:] - good_frames[:-1]

    break_idxs = nx.where(f_diff > max_break)
    break_idxs = break_idxs[0] # hmm, why must I do this?

    start_frame = good_frames[0]
    results = []
    for break_idx in break_idxs:
        stop_frame = good_frames[break_idx]
        
        if (stop_frame - start_frame + 1) >= min_len:
            results.append( (start_frame, stop_frame) )

        # for next loop
        start_frame = good_frames[break_idx+1]
    return results

def print_clock_diffs(results):
    table = results.root.host_clock_info
    hostnames = list(sets.Set(table.cols.remote_hostname))
    for hostname in hostnames:
        print '--',hostname,'--'
        for row in table.where( table.cols.remote_hostname == hostname ):
            diff = row['stop_timestamp'] - row['remote_timestamp']
            dur = row['stop_timestamp'] - row['start_timestamp']
            print '  %.1f msec (within %.1f msec)'%(diff*1e3,dur*1e3)
        print

if __name__=='__main__':
    results = get_results('DATA20060424_193238.h5',mode='r+')
    #del results.root.exact_movie_info
    #results.close()
    #make_exact_movie_info2(results)
    if 0:
        import FastImage
        for camn in [14]:
            for framenumber in range(1231000,1231800):
                print
                print 'camn',camn
                print 'framenumber',framenumber
                frame,timestamp,rt=get_frame_ts_and_realtime_analyzer_state(results,
                                                                            frame=framenumber,
                                                                            camn=camn,
                                                                            clear_threshold=0.2,
                                                                            diff_threshold=11,
                                                                            )
                print 'timestamp',timestamp
                #print 'frame',frame
                if frame is  None:
                    print 'no reconstructor'
                    continue
                raw_im = FastImage.asfastimage(frame)

                use_roi2=1
                use_cmp=1
                return_first_xy=0
                points = rt.do_work(raw_im, timestamp,
                                    framenumber,
                                    use_roi2,
                                    use_cmp,
                                    return_first_xy)
                for (x, y, area, slope, eccentricity, p1, p2, p3, p4, line_found, slope_found) in points:
                    print 're: x,y,p1,eccentricity',x,y,p1,eccentricity

                print

                data2d = results.root.data2d
                for r in data2d.where( data2d.cols.camn == camn ):
                    if r['frame']==framenumber:
                        print 'orig: x,y,p1,eccentricity',r['x'],r['y'],r['p1'],r['eccentricity']
                
##    if 1:
##        del results.root.exact_movie_info
##        results.close()
##        results = get_results('DATA20051122_224900.h5',mode='r+')
##        make_exact_movie_info2(results)
    if len(sys.argv) > 1:
        save_movie(results)
