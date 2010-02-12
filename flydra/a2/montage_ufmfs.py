from __future__ import with_statement
import motmot.ufmf.ufmf as ufmf_mod
import sys, os, tempfile, re, contextlib, warnings
from optparse import OptionParser
import flydra.a2.auto_discover_ufmfs as auto_discover_ufmfs
import numpy as np
import tables
import flydra.a2.utils as utils
import flydra.analysis.result_utils as result_utils
import subprocess, collections
import flydra.a2.ufmf_tools as ufmf_tools
import flydra.reconstruct
import cherrypy  # ubuntu: install python-cherrypy3
import benu

from tables_tools import openFileSafe

def get_tile(N):
    rows = int(np.ceil(np.sqrt(float(N))))
    cols = rows
    return '%dx%d'%(rows,cols)

def get_config_defaults():
    # keep in sync with usage in main() below
    what = {'show_2d_position': False,
            'show_2d_orientation': False,
            'white_background': False,
            'max_resolution': None,
            }
    default = collections.defaultdict(dict)
    default['what to show']=what
    return default

def make_montage( h5_filename,
                  cfg_filename=None,
                  ufmf_dir=None,
                  dest_dir = None,
                  save_ogv_movie = False,
                  no_remove = False,
                  max_n_frames = None,
                  start = None,
                  stop = None,
                  reconstructor_source = None,
                  movie_fnames = None,
                  movie_cam_ids = None,
                  caminfo_h5_filename = None,
                  ):
    config = get_config_defaults()
    if cfg_filename is not None:
        loaded_cfg = cherrypy._cpconfig.as_dict( cfg_filename )
        for section in loaded_cfg:
            config[section].update( loaded_cfg.get(section,{}) )


    if movie_fnames is None:
        movie_fnames = auto_discover_ufmfs.find_ufmfs( h5_filename,
                                                       ufmf_dir=ufmf_dir,
                                                       careful=True )
    else:
        if ufmf_dir is not None:
            movie_fnames = [ os.path.join(ufmf_dir,f) for f in movie_fnames]

    if len(movie_fnames)==0:
        raise ValueError('no input movies -- nothing to do')

    if dest_dir is None:
        dest_dir = os.curdir
    else:
        if not os.path.exists( dest_dir ):
            os.makedirs(dest_dir)

    # get name of data

    datetime_str = os.path.splitext(os.path.split(h5_filename)[-1])[0]
    if datetime_str.startswith('DATA'):
        datetime_str = datetime_str[4:19]

    workaround_ffmpeg2theora_bug = True

    if reconstructor_source is None:
        reconstructor_source = h5_filename

    reconstructor = flydra.reconstruct.Reconstructor(
                    reconstructor_source)

    if caminfo_h5_filename is not None:
        with openFileSafe( caminfo_h5_filename, mode='r' ) as h5:
            camn2cam_id, tmp = result_utils.get_caminfo_dicts(h5)
            del tmp
    else:
        camn2cam_id = None

    blank_images = {}

    all_frame_montages = []
    for frame_enum,(frame_dict,frame) in enumerate(ufmf_tools.iterate_frames(
        h5_filename, movie_fnames,
        reconstructor = reconstructor,
        movie_cam_ids = movie_cam_ids,
        white_background=config['what to show']['white_background'],
        max_n_frames = max_n_frames,
        start = start,
        stop = stop,
        rgb8_if_color = True,
        camn2cam_id = camn2cam_id,
        )):
        tracker_data = frame_dict['tracker_data']

        if (frame_enum%100)==0:
            print '%s: frame %d'%(datetime_str,frame)

        saved_fnames = []
        for movie_idx,ufmf_fname in enumerate(movie_fnames):
            try:
                frame_data = frame_dict[ufmf_fname]
                cam_id = frame_data['cam_id']
                camn = frame_data['camn']
                image = frame_data['image']
                del frame_data
            except KeyError:
                # no data saved (frame skip on Prosilica camera?)
                if movie_cam_ids is not None:
                    cam_id = movie_cam_ids[movie_idx]
                else:
                    cam_id = ufmf_tools.get_cam_id_from_ufmf_fname(ufmf_fname)
                camn = None
                if cam_id not in blank_images:
                    # XXX should get known image size of .ufmf
                    image = np.empty((480,640),dtype=np.uint8); image.fill(255)
                    blank_images[cam_id] = image
                image = blank_images[cam_id]
            save_fname = 'tmp_frame%07d_%s.png'%(frame,cam_id)
            save_fname_path = os.path.join(dest_dir, save_fname)

            pixel_aspect = config[cam_id].get('pixel_aspect',1)
            transform = config[cam_id].get('transform','orig')

            if config['what to show']['max_resolution'] is not None:
                fix_w, fix_h = config['what to show']['max_resolution']
                fix_aspect = fix_w/float(fix_h)
                desire_aspect = image.shape[1]/float(image.shape[0]*pixel_aspect)
                if desire_aspect >= fix_aspect:
                    # image is wider than resolution given
                    device_w = fix_w
                    device_h = fix_w/desire_aspect
                    device_x = 0
                    device_y = (fix_h-device_h)/2.0
                else:
                    # image is taller than resolution given
                    device_h = fix_h
                    device_w = fix_h*desire_aspect
                    device_y = 0
                    device_x = (fix_w-device_w)/2.0
            else:
                device_x = 0
                device_y = 0
                device_w = image.shape[1]
                device_h = int(image.shape[0]*pixel_aspect) # compensate for pixel_aspect
                fix_w = device_w
                fix_h = device_h

            canv=benu.Canvas(save_fname_path,fix_w,fix_h)
            device_rect = (device_x,device_y,device_w,device_h)
            user_rect = (0,0,image.shape[1],image.shape[0])
            with canv.set_user_coords(device_rect, user_rect,
                                      transform=transform):
                canv.imshow(image,0,0)
                if config['what to show']['show_2d_position'] and camn is not None:
                    cond = tracker_data['camn']==camn
                    this_cam_data = tracker_data[cond]
                    xarr = np.atleast_1d(this_cam_data['x'])
                    yarr = np.atleast_1d(this_cam_data['y'])
                    canv.scatter(xarr, yarr,
                                 color_rgba=(0,0,0,1),
                                 radius=10,
                                 )
                    canv.scatter(xarr+1, yarr+1,
                                 color_rgba=(1,1,1,1),
                                 radius=10,
                                 )
                if config['what to show']['show_2d_orientation'] and camn is not None:
                    cond = tracker_data['camn']==camn
                    this_cam_data = tracker_data[cond]
                    xarr = np.atleast_1d(this_cam_data['x'])
                    yarr = np.atleast_1d(this_cam_data['y'])
                    slope = np.atleast_1d(this_cam_data['slope'])
                    thetaarr = np.arctan(slope)
                    line_len = 30.0
                    xinc = np.cos(thetaarr)*line_len
                    yinc = np.sin(thetaarr)*line_len/float(pixel_aspect)
                    for x,y,xi,yi in zip(xarr,yarr,xinc,yinc):
                        canv.plot([x-xi,x+xi],[y-yi,y+yi],
                                  color_rgba=(0,1,0,0.4),
                                  )
                if workaround_ffmpeg2theora_bug:
                    # first frame should get a colored pixel so that
                    # ffmpeg doesn't interpret the whole move as grayscale
                    canv.plot([0,1],[0,1],
                              color_rgba=(1,0,0,0.1),
                              )
                    workaround_ffmpeg2theora_bug = False # Now we already did it.

            canv.save()
            saved_fnames.append( save_fname_path )

        target = os.path.join(dest_dir, 'movie%s_frame%07d.jpg'%(
            datetime_str,frame_enum+1 ))
        tile = get_tile( len(saved_fnames) )
        imnames = ' '.join(saved_fnames)
        # All cameras saved for this frame, make montage
        CMD=("montage %s -mode Concatenate -tile %s -bordercolor white "
             "-title '%s frame %d' "
             "-border 2 %s"%(imnames, tile, datetime_str, frame, target))
        #print CMD
        subprocess.check_call(CMD,shell=True)
        all_frame_montages.append( target )
        if not no_remove:
            for fname in saved_fnames:
                os.unlink(fname)
    print '%s: %d frames montaged'%(datetime_str,len(all_frame_montages),)

    if save_ogv_movie:
        orig_dir = os.path.abspath(os.curdir)
        os.chdir(dest_dir)
        try:
            CMD = 'ffmpeg2theora -v 10 movie%s_frame%%07d.jpg -o movie%s.ogv'%(
                datetime_str,datetime_str)
            subprocess.check_call(CMD,shell=True)
        finally:
            os.chdir(orig_dir)

        if not no_remove:
            for fname in all_frame_montages:
                os.unlink(fname)

def main():
    # keep default config file in sync with get_config_defaults() above
    usage = """%prog DATAFILE2D.h5 [options]

The default configuration correspondes to a config file:

[what to show]
show_2d_position = False
show_2d_orientation = False
white_background =  False
max_resolution = None

Config files may also have sections such as:

[cam7_1]
pixel_aspect=2 # each pixel is twice as wide as tall
transform='rot 180' # rotate the image 180 degrees (See transform
                    # keyword argument of
                    # :meth:`flydra.a2.benu.Canvas.set_user_coords`
                    # for all possible transforms.)

"""

    parser = OptionParser(usage)

    parser.add_option("--dest-dir", type='string',
                      help="destination directory to save resulting files")

    parser.add_option("--ufmf-dir", type='string',
                      help="directory with .ufmf files")

    parser.add_option("--config", type='string',
                      help="configuration file name")

    parser.add_option("--max-n-frames", type='int', default=None,
                      help="maximum number of frames to save")

    parser.add_option("--start", type='int', default=None,
                      help="start frame")

    parser.add_option("--stop", type='int', default=None,
                      help="stop frame")

    parser.add_option("--ogv", action='store_true', default=False,
                      help="export .ogv video")

    parser.add_option('-n', "--no-remove", action='store_true', default=False,
                      help="don't remove intermediate images")

    parser.add_option('--movie-fnames', type='string', default=None,
                      help="names of movie files (don't autodiscover from .h5)")

    parser.add_option('--movie-cam-ids', type='string', default=None,
                      help="cam_ids of movie files (don't autodiscover from .h5)")

    parser.add_option(
        "-r", "--reconstructor",type='string',
        help="calibration/reconstructor path")

    parser.add_option( "--caminfo-h5-filename", type="string",
                       help="path of h5 file from which to load caminfo")

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    movie_fnames = options.movie_fnames
    if movie_fnames is not None:
        movie_fnames = movie_fnames.split( os.pathsep )

    movie_cam_ids = options.movie_cam_ids
    if movie_cam_ids is not None:
        movie_cam_ids = movie_cam_ids.split( os.pathsep )

    h5_filename = args[0]
    make_montage( h5_filename,
                  cfg_filename = options.config,
                  ufmf_dir = options.ufmf_dir,
                  dest_dir = options.dest_dir,
                  save_ogv_movie = options.ogv,
                  no_remove = options.no_remove,
                  max_n_frames = options.max_n_frames,
                  start = options.start,
                  stop = options.stop,
                  reconstructor_source = options.reconstructor,
                  movie_fnames = movie_fnames,
                  movie_cam_ids = movie_cam_ids,
                  caminfo_h5_filename = options.caminfo_h5_filename,
                  )
