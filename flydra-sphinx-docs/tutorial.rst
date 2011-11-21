Tutorial
========

recent build of Flydra documentation (from sources): http://192.168.1.1/flydra_doc/

For general reference (and to make your own contributions to
documentation), the lab wiki is http://vehicles.caltech.edu/wiki/ which
*anyone* can read but requires Caltech IMSS credentials for write
access.  In particular I started a page with very brief descriptions
of handy shell scripts,
http://vehicles.caltech.edu/wiki/doku.php?id=local_flydra_scripts and
a description of the LAN setup,
http://vehicles.caltech.edu/wiki/doku.php?id=lab_network_setup

There is also an *internal* lab website, http://192.168.1.1/ , or
http://flydra.local/ if Zeroconf is running properly. Any
documentation that you do not want to be visible outside the lab (yet)
can go there.


Hardware and gross structural notes
-----------------------------------

Though all blade computers have host files configured to recognize
names like "flyblade13" (as referred to in the text below), most other
computers do not.  Zeroconf is configured on the relevant blades such
that they can be accessed using the usual scheme, HOSTNAME.local.  For
example, flyblade13 (the "mainbrain"; see below) can be found by
flyblade13.local from any computer on the LAN.  Indeed, assuming the
ROS Master is started as instructed below, you will need to set the
environment variable ::

  ROS_MASTER_URI=http://flyblade13.local:11311

In some (rare) cases, you may need to access these directly by IP
address.  The naming scheme is flybladeN has IP address 192.168.1.N.
E.g., flyblade13 has IP address 192.168.1.13

At your level of interacting with Flydra, the two basic computation
points to be concerned with are the "camera nodes" (or "camnodes" for
short) and "Flydra mainbrain" (or just "mainbrain").  Each camnode
runs on a dedicated blade computer, which has a direct and dedicated
gigabit connection to a camera.  There are five cameras, hence five
corresponding computers: flyblade08 through flyblade12.  Mainbrain
provides point tracking and actually publishes to the ROS topic that
you will access.  The mainbrain computer is flyblade13.  It is also
the primary command point, in that you will start up the camnodes,
ptpd, etc. using several shell scripts on it (without having to
explicitly login to the other computers).

All Flydra programs should be run as the user "cowgirl".  This user
can also act as root, i.e. the password for cowgirl is the same as for
root... so be careful if you decide to use "sudo" powers, and *please*
contact me *before* doing so, unless there is an emergency, etc.


From boot to ready
------------------

Generally speaking, you should only rarely need to enter cowgirl's,
i.e. the admin, password to start-up and manage Flydra.  Two key
preliminary steps to be able to run the tracking system is to remote
mount (sshfs) the source directory from flyblade13 (mainbrain) on
flyblade08-12 (camnodes), and to run PTP (precision time protocol) on
all blades.  You must also start roscore (which runs the ROS Master
and Parameter servers).  That is, *after rebooting the machines*:

1. login to flyblade13 as cowgirl;
#. cd to ~/local_flydra_scripts;
#. run script ``node_sshfs.sh`` to remote mount the sources;
#. run ``tictoc.sh`` to start ptp daemons on camnodes and mainbrain computer;
#. run ``killtic.sh`` to see if starting ptpd across the blades succeeded;
   the purpose of ``killtic.sh`` is to kill these processes, but if you
   just choose the default option ("N" for "no, do not kill") it
   provides a quick way to see what ptp processes are running on the
   camnode and mainbrain computers;
#. start roscore; this can take several forms, e.g.

   a. simply open a separate terminal, type ``roscore``, and leave it alone;
   #. type ``screen -d -m roscore`` which runs roscore under a simulated
      terminal (provided by the "screen" program) and immediately
      detaches it; see the manpages on screen for usage.

#. if ROS is setup correctly, then at this point you should be able to
   see the baseline topics, /rosout and /rosout_agg; e.g., run
   ``rostopic list`` from the computer on which you plan to access the
   tracked points.

If you do not have reason to reboot the machines, you can still
perform pieces of the above steps to see if these preliminary items
are already done.  E.g., use ``killtic.sh`` directly to verify ptp daemons
are present, or run rostopic to (effectively) ping the ROS Master.

Finally, and this should not be a problem, but it is mentioned here
just in case, most of flydra and some of its dependencies are
installed in a Python virtual environment (obtained from the program
`virtualenv <http://www.virtualenv.org/>`_).  When you log into
flyblade13, the shell prompt should be prefixed with "(PY)",
indicating that particular virtual environment is active.  If for some
reason you lose this, then enter::

  $ source PY/bin/activate

The shell prompt should then be prefixed by "(PY)".  If for some reason you need to leave this state, then use::

  $ deactivate

to "deactivate" the virtual environment.


Starting Flydra
---------------

As stated earlier, every step is done on (or rather, every command is
executed on) flyblade13.  If you are working remotely, then you will
need X-forwarding to see the mainbrain application GUI.  Note that it
(and many other components of Flydra, not that you'll need them...)
uses OpenGL, which means ssh from a GNU/Linux workstation will be
easiest.  If you somehow get OpenGL forwarding on a Mac, then please
let me know how.

1. enter "flydra_mainbrain" to start the mainbrain application; it is
   common to change the niceness (i.e. scheduling priority) of the
   mainbrain process to max (-19) or near it (e.g., -11); to do this,
   e.g. for -19 niceness, look for the PID in the GUI window title or
   in the terminal from which you started flydra_mainbrain; e.g., if
   the PID is 1337, then you type::

      $ sudo renice -19 1337

#. on flyblade13, there is a collection of flydra-management-related
   shell scripts under /home/cowgirl/local_flydra_scripts; cd to this
   directory and run "./start_nodes.sh log"; this will start a camnode
   program on each camera computer, and (due to the "log" option)
   enable logging of terminal output; the logging is not necessary
   and, by omitting that command-line argument, can be left off;

#. at the time of writing, the center camera (attached to flyblade10)
   is unavailable; thus the result of the last step will be four
   camnodes attaching to the mainbrain process, and being visible as 4
   cameras on the "Camera Preview/Settings" tab of the mainbrain
   application GUI; only two steps remain:

   a. Click the "Load calibration file..." button, and select the file
      "currentCalXML", which is located in the directory /home/cowgirl
      (...likely the directory you'll be in when the find-file dialog
      opens);
   #. Click the "Synchronize cameras" button.  The meaning of both of
      these should be obvious.  Watch the terminal in which you
      started the mainbrain application for status messages.

   Flydra tracking should now be active.  To verify this, check the
   local ros topics for the "mainbrain super packets" by entering::

     $ rostopic list

   where you should see the topic /flydra_mainbrain_super_packets .
   To get a dump of tracked objects, type::

     $ rostopic echo /flydra_mainbrain_super_packets

   (type Ctrl-C to exit when you're done.)


Accessing the data over ROS (setup and example)
-----------------------------------------------

I assume your ROS installation is operational and that you have basic familiarity with the ROS framework... at least as a user.  To subscribe to the flydra topic, you will first need to install the message type, defined in the package ros_flydra ( https://github.com/astraw/ros_flydra ), which can be obtained via ::

  $ git clone https://github.com/astraw/ros_flydra.git

Be sure that you download this copy in (or move to) a ROS-visible directory (i.e. one under or on ROS_PACKAGE_PATH).  Build it with the usual ::

  $ rosmake ros_flydra

In lieu of a demo, I refer you instead to a specific example.  I created a ROS package called flydra_viz ( https://github.com/slivingston/flydra_viz ), which reads objects positions from the mainbrain and publishes them as visualization markers; these can then be used by (i.e. subscribed to and displayed as small spheres) the application rviz ( http://www.ros.org/wiki/rviz ).  To install flydra_viz try ::

  $ git clone https://github.com/slivingston/flydra_viz.git
  $ rosmake flydra_viz

To see what data is available in the mainbrain tracking messages, look
in the ros_flydra package at the definitions under the msg directory.


Notes
-----

root password for all flydra-related blades is written on side of cPCI
container (see rack mount in far corner of lab).

When commands to be entered in a shell are stated, they will tend to
be in quotes "" or on a separate line beginning with the symbol $ (a
very common convention...).
