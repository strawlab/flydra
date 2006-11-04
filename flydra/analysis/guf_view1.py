from PyOSG import osg, osgDB
import OpenGL
import os, math, glob
import pickle

import cgtypes

import fsee
import fsee.Observer

import numpy
import scipy.io
import pylab
import fsee.eye_geometry.util
import result_browser

def get_world(fname):

    scale=1000.0 # convert model from meters to mm
    world_node = osg.MatrixTransform()
    world_node.setMatrix( osg.Matrix.scale(scale,scale,scale) )
    model_node = osgDB.readNodeFile(fname)
    if model_node is None:
        raise ValueError('did not load world model')
    world_node.addChild(model_node)

    stateset = osg.StateSet()
    stateset.setMode(OpenGL.GL.GL_LIGHTING,osg.StateAttribute.OFF)
    world_node.setStateSet(stateset)
    return world_node

def main():

    world_model = get_world(os.path.join(
        fsee.data_dir,'models/WT1/WT1.osg'))

    filename = 'DATA20060315_170142.h5'
    results = result_browser.get_results(filename,mode='r')
    smooth_data = results.root.smooth_data
    
    fstart = 993900
    fend = 994040
    all_frames = range(fstart,fend+1,1)
    
    hz = 100.0
    dt = 1/hz
    done = False
    vision = None
    
    results = {}
    
    if 1:
        if 1:
            fname = 'guf_output'
            print 'saving',fname
            vision = fsee.Observer.Observer(world_node=world_model,
                                            hz=hz,
                                            use_skybox=False,
                                            #use_skybox=True,
                                            full_spectrum=True,
                                            )
            t = -dt
            count = 0
            while count<len(all_frames):
                frame=all_frames[count]
                t+=dt
                count += 1
                print 'count',count

                cur_pos = None
                cur_ori = None
                for row in smooth_data.where( smooth_data.cols.frame == frame ):
                    if cur_pos is not None:
                        raise RuntimeError('more than 2 values for position?!')
                    if cur_ori is not None:
                        raise RuntimeError('more than 2 values for orientation?!')
                    cur_pos = cgtypes.vec3( row['x'], row['y'], row['z'] )
                    cur_ori = cgtypes.quat( row['qw'], row['qx'], row['qy'], row['qz'] )
                vision.step(cur_pos,cur_ori)
                if 0:
                    vision.save_last_environment_map(fname+'%05d.ppm'%count)

                EMDs = vision.get_last_emd_outputs()
                R = vision.get_last_retinal_imageR()
                G = vision.get_last_retinal_imageG()
                B = vision.get_last_retinal_imageB()
                
                vision.save_last_environment_map('envmap%03d.png'%count)
                
                results.setdefault('position_vec3_xyz',[]).append(
                    tuple(cur_pos))
                results.setdefault('orientation_quat_wxyz',[]).append(
                    (cur_ori.w,cur_ori.x,cur_ori.y,cur_ori.z))

                results.setdefault('R',[]).append(R)
                results.setdefault('G',[]).append(G)
                results.setdefault('B',[]).append(B)
                results.setdefault('EMDs',[]).append(EMDs)
    if 1:
        for key in results.keys():
            # make sure scipy can save it
            results[key] = numpy.asarray(results[key])#,dtype=numpy.Float64)
        scipy.io.savemat('guf_data',results)
        done = True
if 1:
    main()
