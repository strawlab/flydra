= Get the !MainBrain GUI running =

== Starting the Precise Time Protocol Daemon (PTPd) ==

On the mainbrain (assuming it is running NTP):
{{{
# check NTP is running
ntpq -p
# There should be some output here indicating at least one NTP server with some offset and jitter.
# (If there is an error or no servers are listed, see below.) (Also of interest is the output 
#  of "ntpdc -c kern").

# Run PTPd and disable frequency scaling:
# (The options below mean: identifier NTP, stratum 2, do not adjust system clock.)
sudo -s -H
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor 
killall ptpd
nice -n -19 ptpd -s2 -i NTP -t
}}}

Doug/humdra - use the following instead of the last 2 lines (because your LAN is on eth1):
{{{
killall ptpd
nice -n -19 ptpd -b eth1 -s2 -i NTP -t
}}}


If NTP is not running, or the "ntpq -p" command returns errors, restart it with:
{{{
sudo /etc/init.d/ntp restart
}}}

On the camera computers

{{{
# run PTPd and disable frequency scaling
sudo -s -H

# This next 2 lines are only necessary if your computer supports 
# frequency scaling (if you have more than 2 cores, repeat accordingly).
echo "performance" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor

killall ptpd
nice -n -19 ptpd
}}}

Doug/humdra - use the following instead of the last 2 lines (because your LAN is on eth1):
{{{
killall ptpd
nice -n -19 ptpd -b eth1
}}}


On any computer, you can ensure that PTPd is running by checking the list of running processes:
{{{
ps aux | grep ptpd
# There should be a line indicating a running ptpd process.
# (Be careful not to be confused by your "grep ptpd" process.)
}}}

== Running the !MainBrain GUI ==

On the main brain computer, run:

{{{
# You probably do NOT want to run this as root.
flydra_mainbrain
}}}
== Running the camera nodes ==

Once the Main Brain GUI is up and running, run the following on the camera node computers:
{{{
# You probably do NOT want to run this as root.
./run_cam.sh
# (This should call "flydra_camera_node" with the appropriate options).
}}}

Note that some of old Basler cameras have broken firmware and require a trigger mode "2" to trigger properly.

(You may also be interested in some command line options. See {{{flydra_camera_node --help}}} for more information.

== OLD - Running the camera nodes ==
'''Ignore this section -- it is old.'''
Once the GUI is up and running, run the following on the camera node computers:

Prosilica cameras:
{{{
# If your camera is at IP address of 192.168.1.51, you can 
# set the packet size on the command line with:
CamAttr -i 192.168.1.51 -s PacketSize 1500
CamAttr -i 192.168.1.51 -s StreamBytesPerSecond 123963084

# Now start the camera grabbing program:
shmwrap_prosilica_gige
}}}

Basler (or other 1394) cameras:
{{{
# To reset the 1394 bus (necessary if you get the "Could not allocate bandwidth" error):
dc1394_reset_bus

# Now start the camera grabbing program:
shmwrap_dc1394
}}}

Now that the shared-memory camera grabber is running, run this:
{{{
flydra_camera_node --wrapper sharedmem --backend sharedmem --num-points=2
}}}

= Load the Camera Configurations =

'''This is currently broken. Don't do this right now -- change the settings by hand.''' In {{{File->Open Camera Configuration}}}, open an appropriate configuration.

= Load a calibration =

Press the {{{Load Calibration...}}} button. Select an appropriate calibration.

= Make sure cameras are synchronized =

(As a prerequisite, the cameras must be in external trigger mode. This is usually Trig mode: "1" in the per-camera configuration in the main brain.)

Press the "Synchronize cameras" button. (Or, if you don't have a working Flydra Trigger Device, unplug your function generator for > 1 second.)

= Basic operation =

The green dots are tracked in 2D on the local cameras.

The red dots are the 3D reconstruction back-projected into the camera view.

The ongoing background estimate can be cleared (set to zero) on individual cameras by pressing the appropriate GUI button, or on all cameras by pressing the {{{<C>}}} key. The estimate can be set to the current image by doing {{{take}}} (or pressing the {{{<T>}}} key for all cameras). (Note: I always press {{{<C>}}} then {{{<T>}}} because I think there may be a small bug when just {{{<T>}}} is pressed.)



= No longer useful for MainBrain =

== Getting the mainbrain computer able to talk to the trigger device ==

Prevent the linux kernel from de-powering the USB trigger device (or any USB device):
{{{
# login as root:
sudo -s -H
# (enter password)

echo -n -1 > /sys/module/usbcore/parameters/autosuspend
}}}

