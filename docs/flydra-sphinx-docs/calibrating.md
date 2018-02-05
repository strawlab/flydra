Flydra Camera Calibration Workflow
==================================

Calibrate from top-to-bottom:

### Step 0: turn on cameras and get viewers running

Do this:

    roslaunch ./flycube_only_flydra.launch --screen

### Step 1: setup cameras (zoom, focus, aperture, gain) and lights

Setup camera position, zoom, focus (thanks to an object in the flycube) and aperture (slightly under the maximum).
The shutter times and gains are already specified in the flydra.yaml file (located in flycave/launch/flycubeX/).
The gains could be improved.

View the luminosity distribution of each camera (replace the camera names accordingly):

    rosrun ros_flydra camhistograms --camera /Basler_xxx/image_raw etc.

Try to have nicely widespread luminosity distributions. Look at those distributions while the room
lights are off! Write the chosen gain values in flydra.yaml.

### Step 2: remove the distortion errors of the cameras

Run this (change camera name accordingly):
    rosrun camera_calibration cameracalibrator.py --size 6x8 --square=0.029  image:=/Basler_21020232/image_raw camera:=Basler_21020232

Show a checkerboard to the camera (the checkerboard parameters are hardcoded into the operator-console as a 8x6 checkerboard with 29 mm squares). Try to show the checkerboard at different distances and angles to increase the four gauges to almost the maximum. Do not forget to show the checkerboard corners in the corners of the camera field of view. Once the four gauges are filled, click "calibrate", then wait. Then click on "save", wait a bit, and finally click on "commit".

Verify the removal of distortion errors like this:

Keep the `flycube_only_flydra.launch` file running in the first tab.

`ROS_NAMESPACE=Basler_xxx rosrun image_proc image_proc` in a second tab to launch the image\_proc node (this loads the intrinsic calibrations from ~/.ros/camera_info and generates undistorted images),

`rosrun image_view image_view image:=/Basler_xxx/image_rect_color` in a third tab to launch the viewer node.


### Step 3: calibrate flydra

Room lights should be off.

In a first tab, start the flydra (tracking system) calibration with (enter the right flycube name):

    roscd flycave/launch/flycubeX/
    roslaunch ./flycube_only_flydra_calib.launch --screen

In a second tab, synchronize the cameras:

    roscd ros_flydra
    python scripts/simple-console.py --sync

Then start saving data and begin to collect data by moving a LED in the arena (try move the LED in the whole arena volume, also turn it off and on sometimes to validate synchronization). When you are done, stop saving:

    roscd ros_flydra
    python scripts/simple-console.py --start-saving-hdf5

    # move your LED around....

    python scripts/simple-console.py --stop-saving-hdf5

Now, in a third tab, we will run the "Multi Cam Self Cal" (MCSC). The file created by saving the moving LED is in the folder ~/DATA. Let's pretend it is called `20130726_122220.mainbrain.h5`. Do this:

    cd ~/DATA
    export DATAFILE=20130726_122220.mainbrain.h5
    flydra_analysis_generate_recalibration --2d-data ${DATAFILE} --disable-kalman-objs ${DATAFILE} --undistort-intrinsics-yaml=${HOME}/.ros/camera_info  --run-mcsc --use-nth-observation=4

Try to have about 500 points in your calibration. (Adjust the `--use-nth-observation=N` option
in the above call if necessary.) A good calibration results in a mean reprojection error of 0.5 pixels
or less.

Your MCSC result is in the directory `~/DATA/${DATAFILE}.recal/result`. Retrack your calibration data:

    flydra_kalmanize ${DATAFILE} -r ${DATAFILE}.recal/result
    export DATAFILE_RETRACKED=`python -c "print '${DATAFILE}'[:-3]"`.kalmanized.h5
    flydra_analysis_plot_timeseries_2d_3d ${DATAFILE} -k ${DATAFILE_RETRACKED} --disable-kalman-smoothing

The last command line opens a plot on which you should see very nice alignment of the 3D reprojected trajectories and the 2D data. The "3d error meters" should not exceed 0.045.

Move the three generated files from ~/DATA to flycave/calibration/flycubeX. You can also save here the above generated plot and the mean reprojection error value in a text file.

Now you have a working calibration, which is NOT aligned or scaled to the flycube coordinate system, but is able to track. Scaling can be quite important for good tracking.

TODO: describe how to align and scale the calibration using the alignment GUI.


