import os, pickle
import numpy

def convert_to_multicamselfcal_format(cam_ids,data):
    n_cams = len(cam_ids)
    n_pts = len(data)

    IdMat = numpy.zeros( (n_cams, n_pts), dtype=numpy.uint8 )
    points = numpy.nan*numpy.ones( (n_cams*3, n_pts), dtype=numpy.float )

    cam_id2idx = {}
    for i,cam_id in enumerate(cam_ids):
        if cam_id in cam_id2idx:
            raise ValueError('cam_id already in cam_id2idx')
        cam_id2idx[cam_id]=i

    # fill data
    for col_num,row_data in enumerate(data):
        for cam_id, point2d in row_data:
            row_num = cam_id2idx[cam_id]

            IdMat[row_num,col_num]=1
            points[row_num*3,  col_num]=point2d[0]
            points[row_num*3+1,col_num]=point2d[1]
            points[row_num*3+2,col_num]=1.0

    return IdMat, points

def save_ascii_matrix(filename,m):
    fd=open(filename,mode='wb')
    for row in m:
        fd.write( ' '.join(map(str,row)) )
        fd.write( '\n' )

def do_save_calibration_data(calib_dir, cam_ids, data_to_save, Res):
    if 0:
        # temporarily save all data as pickle
        fd = open(os.path.join(calib_dir,'cal_data_temp_format.pkl'),mode='wb')
        pickle.dump( data_to_save, fd )
        fd.close()

    IdMat,points = convert_to_multicamselfcal_format(cam_ids,data_to_save)

    save_ascii_matrix(os.path.join(calib_dir,'IdMat.dat'),IdMat)
    save_ascii_matrix(os.path.join(calib_dir,'points.dat'),points)

    # save extra data

    Res = numpy.array( Res )
    save_ascii_matrix(os.path.join(calib_dir,'Res.dat'),Res)

    fd = open(os.path.join(calib_dir,'camera_order.txt'),'w')
    for cam_id in cam_ids:
        fd.write('%s\n'%cam_id)
    fd.close()
