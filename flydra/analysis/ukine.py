import glob, os, sys
import FlyMovieFormat
import matplotlib
matplotlib.use('GTKAgg') # TkAgg doesn't work, at least without ioff(), which I haven't tried
import pylab
import tables
import flydra.reconstruct
import numpy

# base file names
if 0:
    base_fname = 'full_20060315_202300_%s_bg.fmf'
    cal_source = 'DATA20060315_170142.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20060515_190909_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060515_190905.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20060516_191746_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060516_194920.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20060717_185549_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060717_185535.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')

    
elif 0:
    base_fname = 'full_20060830_184358_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060830_184701.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20060724_173446_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060724_173517.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20060915_171705_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20060915_173304.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20061129_125343_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20061127_193406.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 0:
    base_fname = 'full_20061205_193159_%s_bg.fmf'
    # hdf5 file containing calibration data
    cal_source = 'DATA20061205_193629.h5'
    reconstructor_source = tables.openFile(cal_source,mode='r')
elif 1:
    
    if 0:
        # all files in current directory
        files = os.listdir(os.curdir)
        fmfs = [f for f in files if f.startswith('full') and f.endswith('bg.fmf')]
        print 'fmfs',fmfs
        if 0:
            cals = [f for f in files if f.startswith('DATA') and f.endswith('.h5')]
        elif 0:
            cals = [f for f in files if f.startswith('cal_')]
        elif 1:
            cal_source = sys.argv[1]
            cals = [os.path.expanduser(cal_source)]
            
        cam_ids = []
        fname_by_cam_id = {}
        for full_fmf in fmfs:
            dirname,fmf = os.path.split(full_fmf)
            print fmf
            fmf = fmf.replace('_bg.fmf','')
            spl = fmf.split('_')
            start = '_'.join(spl[:3])
            cam_id = '_'.join(spl[3:])
            print 'fmf',fmf
            print 'start',start
            print
            
            fname_by_cam_id[cam_id] = full_fmf
            cam_ids.append( cam_id )
            
    elif 1:
        # fmf files in ~/camN/FLYDRA_LARGE...
        cal_source = sys.argv[1]
        date_time = sys.argv[2]

        cam_computers = ['cam1','cam2','cam3','cam4','cam5']

        fname_by_cam_id = {}
        for cam in cam_computers:
            dirname = os.path.expanduser('~/%s/FLYDRA_LARGE_MOVIES/'%(cam,))
            fnames = os.listdir( dirname )
            this_fmfs = {}
            cam_id = None
            for fname in fnames:
                if date_time in fname:
                    if fname.endswith('_bg.fmf'):
                        this_fmfs['bg'] = fname
                    elif fname.endswith('_std.fmf'):
                        this_fmfs['std'] = fname
                    else:
                        tmp_dirname, tmp_fname = os.path.split(fname)
                        tmp_fname = os.path.splitext(tmp_fname)[0]
                        cam_id = '_'.join(tmp_fname.split('_')[-2:])
                        this_fmfs['single'] = fname
            preferred_order = ['bg','single','std']
            for name in preferred_order:
                if name in this_fmfs:
                    fname_by_cam_id[cam_id] = os.path.join(dirname,this_fmfs[name])
                    break
                
        cals = [cal_source]

        print cals
        print fname_by_cam_id
        
####################################        

        
    print cals

    if len(cals)>1:
        print cals
        raise ValueError('too many calibration files')
    elif len(cals)==0:
        raise ValueError('no calibration file')
    
    # hdf5 file containing calibration data
    cal_source = cals[0]
    if cal_source.endswith('.h5'):
        reconstructor_source = tables.openFile(cal_source,mode='r')
    else:
        reconstructor_source = cal_source

recon = flydra.reconstruct.Reconstructor(reconstructor_source)

class ClickGetter:
    def on_click(self,event):
        # get the x and y coords, flip y from top to bottom
        x, y = event.x, event.y
        if event.button==1:
            if event.inaxes is not None:
                print >> sys.stderr, 'data coords (distorted)', event.xdata, event.ydata
                self.coords = event.xdata, event.ydata

click_locations = []
for cam_id in recon.cam_ids:
    fname = fname_by_cam_id[cam_id]
    print >> sys.stderr, cam_id,fname
    
    fmf = FlyMovieFormat.FlyMovie(fname)
    frame,timestamp = fmf.get_frame(0)
    fmf.close()
    
    pylab.imshow(frame,origin='lower')

    cg = ClickGetter()
    binding_id=pylab.connect('button_press_event', cg.on_click)
    pylab.show()
    pylab.disconnect(binding_id)
    if hasattr(cg,'coords'):
        # user clicked
        click_locations.append( (cam_id, recon.undistort(cam_id,cg.coords) ))
    print >> sys.stderr

X = recon.find3d( click_locations, return_line_coords = False )

l2norms = []
for cam_id,orig_2d_undistorted in click_locations:
    predicted_2d_undistorted = recon.find2d( cam_id, X )
    o = numpy.asarray(orig_2d_undistorted)
    p = numpy.asarray(predicted_2d_undistorted)
    l2norm = numpy.sqrt(numpy.sum((o-p)**2))
    print >> sys.stderr, '%s (% 5.1f, % 5.1f) (% 5.1f,% 5.1f) % 5.1f'%(cam_id,
                                                        orig_2d_undistorted[0],
                                                        orig_2d_undistorted[1],
                                                        predicted_2d_undistorted[0],
                                                        predicted_2d_undistorted[1],
                                                        l2norm)
    l2norms.append( l2norm )

print >> sys.stderr
print >> sys.stderr, 'mean reconstruction error:',numpy.mean(l2norms)
print >> sys.stderr
print 'X=(% 5.1f,% 5.1f,% 5.1f) # 3d location'%(X[0],X[1],X[2])
