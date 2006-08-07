import tables
import tables as PT
import numpy
import os

class KalmanEstimates(PT.IsDescription):
    obj_id     = PT.Int32Col(pos=0,indexed=True)
    frame      = PT.Int32Col(pos=1)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)
    xvel       = PT.Float32Col(pos=5)
    yvel       = PT.Float32Col(pos=6)
    zvel       = PT.Float32Col(pos=7)
    xaccel     = PT.Float32Col(pos=8)
    yaccel     = PT.Float32Col(pos=9)
    zaccel     = PT.Float32Col(pos=10)

class FilteredObservations(PT.IsDescription):
    obj_id     = PT.Int32Col(pos=0,indexed=True)
    frame      = PT.Int32Col(pos=1,indexed=True)
    x          = PT.Float32Col(pos=2)
    y          = PT.Float32Col(pos=3)
    z          = PT.Float32Col(pos=4)

class TrackedObject: pass

filename = 'DATA20060719_180955.tracked.h5'
results = tables.openFile(filename,mode='r')

if 1:
    filename = os.path.splitext(results.filename)[0]+'.resaved-tracked.h5'
    if os.path.exists(filename):
        os.unlink(filename)
    h5file = PT.openFile(filename, mode="w", title="tracked Flydra data file")
    ct = h5file.createTable # shorthand
    root = h5file.root # shorthand
    h5_xhat = ct(root,'kalman_estimates', KalmanEstimates,
                 "Kalman a posteri estimates of tracked object")
    h5_obs = ct(root,'observations', FilteredObservations,
                "observations of tracked object")
    

obj_id = -1
tros = []
while 1:
    obj_id+=1
    try:
        xhats_table = getattr(results.root,'kalman_%d'%obj_id)
    except tables.exceptions.NoSuchNodeError:
        break
    obs_table = getattr(results.root,'observations_%d'%obj_id)

    tro = TrackedObject()
    tro.observations_frames = numpy.asarray(obs_table.read(flavor='numpy',field='frame'))
    if 1:
        obj_id_array = obj_id*numpy.ones(tro.observations_frames.shape,dtype=numpy.int32)
        tro.observations_data = numpy.hstack((numpy.asarray(obs_table.read(flavor='numpy',field='x'))[:,numpy.newaxis],
                                              numpy.asarray(obs_table.read(flavor='numpy',field='y'))[:,numpy.newaxis],
                                              numpy.asarray(obs_table.read(flavor='numpy',field='z'))[:,numpy.newaxis]))

        list_of_obs = [ tro.observations_data[:,i] for i in range(3) ]
        
        names = PT.Description(FilteredObservations().columns)._v_names
        recarray = numpy.rec.fromarrays([obj_id_array,tro.observations_frames]+list_of_obs,
                                        names = names)
        h5_obs.append(recarray)
        h5_obs.flush()

        tro.frames = numpy.asarray(xhats_table.read(flavor='numpy',field='frame'))
        tro.xhats = numpy.hstack((numpy.asarray(xhats_table.read(flavor='numpy',field='x'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='y'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='z'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='xvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='yvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='zvel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='xaccel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='yaccel'))[:,numpy.newaxis],
                                  numpy.asarray(xhats_table.read(flavor='numpy',field='zaccel'))[:,numpy.newaxis]))
        
        obj_id_array = obj_id*numpy.ones(tro.frames.shape,dtype=numpy.int32)
        list_of_xhats = [tro.xhats[:,col] for col in range(9)]
        names = PT.Description(KalmanEstimates().columns)._v_names
        recarray = numpy.rec.fromarrays([obj_id_array,tro.frames]+list_of_xhats,
                                        names = names)
        h5_xhat.append(recarray)
        h5_xhat.flush()


    print 'loaded',obj_id
##    if obj_id==300:
##        break
    
h5file.close()
results.close()
