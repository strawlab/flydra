#!/usr/bin/env python
import Pyro.core, Pyro.errors
import struct
import data_frame

import flydra.reconstruct

from matplotlib.matlab import *
#import matplotlib.numerix as nx
import numarray as nx

from numarray.ieeespecial import nan
import numarray.ieeespecial

Pyro.config.PYRO_MULTITHREADED = 0 # We do the multithreading around here!

Pyro.config.PYRO_TRACELEVEL = 3
Pyro.config.PYRO_USER_TRACELEVEL = 3
Pyro.config.PYRO_DETAILED_TRACEBACK = 1
Pyro.config.PYRO_PRINT_REMOTE_TRACEBACK = 1

Pyro.core.initClient(banner=0)

def get2d(Pmat,X):
    x=nx.matrixmultiply(Pmat,X)
    return (x/x[2])[:2,0]

def load_data(filename2d,filename3d_fast,filename3d_best):

    #########################
    # 2D data
    #########################

    fd = open(filename2d,'rb')
    dats = fd.read()
    SAVE_2D_FMT = '<Bidfffffffff'
    
    sz=struct.calcsize(SAVE_2D_FMT)
    
    n=len(dats)/sz
    
    data = []
    for i in range(n):
        start = i*sz
        stop = i*sz+sz
        tmp = dats[start:stop]
        tmp = struct.unpack(SAVE_2D_FMT,tmp)
        data.append(tmp)
        
    data = nx.array(data,nx.Float64)
    
    cam = data[:,0].astype(nx.UInt8)
    frame = data[:,1].astype(nx.Int32)
    time = data[:,2]
    x = data[:,3]
    y = data[:,4]
    area = data[:,5]
    slope = data[:,6]
    eccentricity = data[:,7]
    p1 = data[:,8]
    p2 = data[:,9]
    p3 = data[:,10]
    p4 = data[:,11]

    value_dict = dict(
      cam = cam,
      frame = frame,
      time = time,
      x = x,
      y = y,
      slope = slope
      )

    fields_order = 'cam', 'frame', 'time', 'x', 'y', 'slope'

    results2d = data_frame.DataFrame(value_dict = value_dict,
                               fields_order = fields_order)

    #########################
    # 3D data
    #########################
    def load_ascii_matrix(filename):
        fd=open(filename,mode='rb')
        buf = fd.read()
        lines = buf.split('\n')[:-1]
        return nx.array([map(float,line.split()) for line in lines])
    results3d_fast = load_ascii_matrix(filename3d_fast)
    results3d_best = load_ascii_matrix(filename3d_best)
    return results2d, results3d_fast, results3d_best

def get_cams(global_data_fname):
    fd = open(global_data_fname,'rb')
    cam_ids = []
    for i,line in enumerate(fd.readlines()):
        cam_id, hack_cam_no, time_offset = line.split()
        cam_ids.append( (cam_id, int(hack_cam_no)) )
    cam_order = ['']*len(cam_ids)
    for cam_id, hack_cam_no in cam_ids:
        cam_order[hack_cam_no]= cam_id
    return cam_order

def get_server(cam_id):
    port = 9888
    hostname = cam_id.split(':')[0]

    URI = "PYROLOC://%s:%d/%s" % (hostname,port,'frame_server')
    print 'connecting to',URI,'...',
    frame_server = Pyro.core.getProxyForURI(URI)
    print '  OK'
    return frame_server

class Results:
    def __init__(self,movie_filename='/tmp/raw_video.fmf'):
        self.cam_ids = get_cams('camera_data.dat')
        self.frame_server = {}
        for cam_id in self.cam_ids:
            self.frame_server[cam_id] = get_server(cam_id)
        self.data2d, self.data3d_fast, self.data3d_best = load_data(
            'raw_data.dat','raw_data_3d_fast.dat','raw_data_3d_best.dat')

        self.frame_offset = {}
        print 'requesting movie',movie_filename
        for i,cam_id in enumerate(self.cam_ids):
            self.frame_server[cam_id].load(movie_filename)
            frame, timestamp = self.frame_server[cam_id].get_frame(0)
            time_frame=self.data2d.where_field_equal('time',timestamp,1e-10)
            if time_frame is None:
                print 'ERROR: timestamp for %s is %s, but not in data2d!'%(cam_id,repr(timestamp))
            cam_frame = time_frame.where_field_equal('cam',i)
            assert len(cam_frame)==1
            frame_offset = cam_frame[0]['frame']
            self.frame_offset[cam_id]=frame_offset
            #print cam_id,'frame_offset',frame_offset

        self.R = flydra.reconstruct.Reconstructor( calibration_dir = '/home/astraw/Cal-2004-11-08' )

        # initialize data structures
        cam_2d_data = {}
        for i, cam_id in enumerate(self.cam_ids):
            #print 'filling dict for %s with absolute_cam_no %d'%(cam_id,i)
            cam_2d_data[cam_id] = self.data2d.where_field_equal('cam',i)

        # find frames in all movies
        first_frame_numbers = []
        first_timestamps = []
        for i, cam_id in enumerate(self.cam_ids):
            frame, timestamp = self.frame_server[cam_id].get_frame(0)
            first_timestamps.append(timestamp)
            row=cam_2d_data[cam_id].where_field_equal('time',timestamp,1e-10)
            first_frame_numbers.append(row[0]['frame'])
        self.first_frame = max(first_frame_numbers)
        first_timestamp = min(first_timestamps)

        last_frame_numbers = []
        last_timestamps = []
        for i, cam_id in enumerate(self.cam_ids):
            frame, last_movie_timestamp = self.frame_server[cam_id].get_frame(-1)
            last_2d_data_timestamp=cam_2d_data[cam_id][-1]['time']
            if last_movie_timestamp <= last_2d_data_timestamp:
                timestamp = last_movie_timestamp
            else:
                timestamp = last_2d_data_timestamp
                print 'WARNING: 2d data is truncated compared to movie for %s'%cam_id
            row=cam_2d_data[cam_id].where_field_equal('time',timestamp,1e-10)
            if row is None:
                print 'ERROR: timestamp for %s is %s, but not in cam_2d_data!'%(cam_id,repr(timestamp))
            assert len(row)==1
            last_timestamps.append(timestamp)
            last_frame_numbers.append(row[0]['frame'])
        self.last_frame = min(last_frame_numbers)
        last_timestamp = max(last_timestamps)

        print 'frame range [%d,%d]'%(self.first_frame,self.last_frame)

        # trim data2d
        self.data2d = self.data2d.where_field_greaterequal('time',first_timestamp)
        self.data2d = self.data2d.where_field_lessequal('time',last_timestamp)

def plot_whole_movie_3d(results, typ='best'):
    if typ == 'fast':
        data3d = results.data3d_fast
    elif typ == 'best':
        data3d = results.data3d_best
    # plot it!
    data3d_start = nx.nonzero(nx.greater_equal( data3d[:,0], results.first_frame ))[0]
    data3d_stop = nx.nonzero(nx.less_equal( data3d[:,0], results.last_frame ))[-1]

    d = data3d[data3d_start:data3d_stop,:]

    f = d[:,0]
    x = d[:,1]
    y = d[:,2]
    z = d[:,3]

    plot(f,x,'r.')
    plot(f,y,'g.')
    plot(f,z,'b.')
    title(typ+' data')
    return locals()

def plot_all_images(results,frame_no,show_raw_image=True,typ='best'):
    if typ == 'fast':
        data3d = results.data3d_fast
    elif typ == 'best':
        data3d = results.data3d_best
    frame_table = results.data2d.where_field_equal('frame',frame_no)
    cams = frame_table.get_all_values('cam')
    assert len(cams) == len(results.cam_ids)

    try:
        Xi = nx.nonzero( nx.equal( data3d[:,0], frame_no ))[0][0]
        print 'Xi',Xi
        print 'data3d.shape',data3d.shape
        X = data3d[Xi,1:4]
        print 'data3d[Xi]',data3d[Xi]
        line3d = data3d[Xi,4:10]
    except IndexError:
        X = None
        line3d = None

    clf()
    cam_ids = results.cam_ids[:]
    cam_ids.sort()
##    for i,cam_id in enumerate(results.cam_ids):
    for subplot_number,cam_id in enumerate(cam_ids):
        i = results.cam_ids.index(cam_id)
        subplot(2, 3, subplot_number+1)
        
        movie_frame = frame_no - results.frame_offset[cam_id]
        if show_raw_image:
            im, timestamp = results.frame_server[cam_id].get_frame(movie_frame)
            imshow(im,origin='lower',interpolation='nearest',cmap=cm.gray)

        cam_table = frame_table.where_field_equal('cam',i)
        x=cam_table[0]['x']
        y=cam_table[0]['y']
        slope=cam_table[0]['slope']

        if x>=0:
            # raw 2D
            lines=plot([x],[y],'o')
            
            if show_raw_image:
                green = (0,1,0)
                set(lines[0],'markerfacecolor',green)
                set(lines[0],'markeredgecolor',green) 	 
                set(lines[0],'markeredgewidth',2)

            if not len(numarray.ieeespecial.getnan(slope)[0]):
                ox0 = x
                oy0 = y
                angle_radians = math.atan(slope)
                r = 20.0
                odx = r*math.cos( angle_radians )
                ody = r*math.sin( angle_radians )

                x0 = ox0-odx
                x1 = ox0+odx
                
                y0 = oy0-ody
                y1 = oy0+ody
                lines=plot([x0,x1],[y0,y1],'-')
                if show_raw_image:
                    green = (0,1,0)
                    set(lines[0],'color',green)
                    set(lines[0],'linewidth',2)

        if X is not None:
            if line3d is None:
                x,y=results.R.find2d(cam_id,X)
                l3=None
            else:
                x,l3=results.R.find2d(cam_id,X,line3d)
                x,y=x
                
            width, height = results.R.get_resolution(cam_id)
            near = 10
            if x>=0-near and x < width+near and y>=0-near and y < height+near:
                # reconstructed 2D
                lines=plot([x],[y],'rx')
                
            if l3 is not None:
                a,b,c=l3
                # ax+by+c=0

                # y = -(c+ax)/b
                # x = -(c+by)/a
                
                x1=0
                y1=-(c+a*x1)/b
                if y1 < 0:
                    y1 = 0
                    x1 = -(c+b*y1)/a
                elif y1 >= height:
                    y1 = height-1
                    x1 = -(c+b*y1)/a
                
                x2=width-1
                y2=-(c+a*x2)/b
                if y2 < 0:
                    y2 = 0
                    x2 = -(c+b*y2)/a
                elif y2 >= height:
                    y2 = height-1
                    x2 = -(c+b*y2)/a
                
                lines=plot([x1,x2],[y1,y2],'r-')
                
        title(cam_id)
        
if __name__ == '__main__':
    try:
        results
    except NameError:
        results = Results()
    if 0:
        for frame in range(3901,4134,2):
            plot_all_images(results, frame, True)
            fname = 'frame%04d.png'%frame
            #fname = 'raw_frame%04d.png'%frame
            print 'saving',fname
            savefig(fname)
            clf()
