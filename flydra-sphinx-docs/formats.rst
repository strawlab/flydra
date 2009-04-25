HDF5 file formats used by Flydra
********************************


Existing data format 
====================

Each data file is named after a timestamp: ``DATAYYYYMMDD_HHMMSS`` 


Table ``data2d_distorted``
--------------------------
 
This is the raw data from the camera.

The fields are as follows: ::

        camn                     ID camera
        frame                    camera frame (relative to camera)
        frame_pt_idx             index of the point tracked on the same frame by this camera
        timestamp                synchronized timestamp
        x                        pixel coordinates    
        y
        cam_received_timestamp   global timestamp ??? (ignore)
        area                     ???      
        slope                    ???
        eccentricity             ???
        cur_val                  ???
        mean_val                 ???
        mean2_val                ???

Notes:

- The ``camn`` field changes from experiment to experiment. To get the id of the actual camera used, look in the
  ``cam_info`` table.

- If there are no points tracked for a particular frame, then a dummy entry is produced, with ``x`` and ``y`` set to ``NaN``.


Table ``kalman_observations``
--------------------------

This table record the maximum likelihood estimates of fly position, given the current frame observations.
(results of data association and transformation 2d->3d).

Note that the name ``kalman_observations`` might be a bit misleading.


The fields are as follows: ::

        obj_id                   
        frame                    
        x                        coordinates
        y
        z 
        obs_2d_idx               ??? 


Table ``kalman_estimates``
--------------------------

This is the data after the operation of Kalman smoothing.

::

        obj_id                   track id 
        frame
        timestamp          
        x                        coordinates (m)
        y
        z
        xvel                     velocities  (m/s)
        yvel
        zvel
        P00 ... P55              covariances



Future data format 
==================


This part describes the current idea about the tables to be added.


Filtered table
--------------

::

	filtered
	
		obj_id           int32
		frame            int64
		timestamp        double64
		
		position                   3x1  vector     
		attitude                   3x3  rotation matrix
		linear_velocity_world      3x1  vector
		linear_velocity_body       3x1  vector
		angular_velocity_world     3x1  vector
		angular_velocity_body      3x1  vector
		torque_body                3x1  vector
		force_body                 3x1  vector
		head_orientation           3x3  rotation matrix
		

Simulated visual input
----------------------

::

    visual_sim           simulated visual data

       obj_id
       frame            
       timestamp

       visual_data               1500 floats       "y"         spatial-filtered   but not temporal
       visual_data_dot           1500 floats       "y_dot"
       mu                        1500 floats
       retinal_velocity          1500 x 2 floats
       emds                      4000 (roughly)    

       lptc                      50   floats       simulated LPTC activity

       visual_data_filt          1500 floats       y filtered according to the eye
       visual_data_dot_filt      1500 floats       ....
       lptc_filt                 50   floats       ....


Debug images
------------

::

	debug_images        images visualizing quantities in the flyscope 
	
       visual_data_vis           image             
       mu_vis                    image             
       optic_flow_vis            image             



    - add label segmentation of the trajectories	

