from __future__ import print_function
import roslib
import roslib.packages
import rospkg

roslib.load_manifest("rospy")
roslib.load_manifest("rosgraph")
import rospy
import rospy.names
import rosgraph.masterapi

import os.path
import string
import sys
import errno
from urlparse import urlparse
import socket


class Log:
    to_console = True
    force_output = None

    def __init__(self, to_ros=False):
        self.to_ros = to_ros

    def _write(self, x, f):
        _file = self.force_output if self.force_output != None else f
        if x:
            while 1:
                try:
                    _file.write(x)
                    if not x.endswith("\n"):
                        _file.write("\n")
                    break
                except IOError as err:
                    if err.args[0] == errno.EINTR:  # interrupted system call
                        continue
        while 1:
            try:
                _file.flush()
                break
            except IOError as err:
                if err.args[0] == errno.EINTR:
                    continue

    def info(self, x):
        if self.to_ros:
            # loginfo prints to stdout
            rospy.loginfo(x)
            return
        if self.to_console:
            self._write(x, sys.stdout)

    def debug(self, x):
        if self.to_ros:
            rospy.logdebug(x)
        if self.to_console:
            self._write(x, sys.stdout)

    def warn(self, x):
        if self.to_ros:
            rospy.logwarn(x)
            # logwarn prints to stderr
            return
        if self.to_console:
            self._write(x, sys.stderr)

    def fatal(self, x):
        if self.to_ros:
            rospy.logfatal(x)
            # logwarn prints to stderr
            return
        if self.to_console:
            self._write(x, sys.stderr)


# FROM motmot_ros_utils.roslib.io
def decode_url(url, required=False):
    """
    Example URL syntax:
        file:///full/path/to/local/file.yaml
        file:///full/path/to/videre/file.ini
        package://camera_info_manager/tests/test_calibration.yaml
        package://ros_package_name/calibrations/camera3.yaml
        /full/path
        ~/home/path

    The file: URL specifies a full path name in the local system. The package: URL is handled
    the same as file:, except the path name is resolved relative to the location of the named ROS
    package, which must be reachable via $ROS_PACKAGE_PATH.

    The following variables can be replaced

        ${NODE_NAME} resolved to the current node name.
        ${ROS_HOME} resolved to the $ROS_HOME environment variable if defined, ~/.ros if not.

    XXXX: Use the ros resource_retriever class here one day, when it has python bindings and
          variable substitution
    """
    if url.startswith("~"):
        url = os.path.expanduser(url)
    elif url.startswith("file://"):
        url = url[7:]
    elif url.startswith("package://"):
        package, fragment = url[10:].split("/", 1)
        url = os.path.join(
            roslib.packages.get_pkg_dir(package, required=False), fragment
        )

    nodename = rospy.names.get_name()
    if nodename:
        # rospy returns rully resolved name, so remove first char (~,/) and
        # return only the first fragment
        nodename = nodename[1:].split(" ")[-1]
    else:
        nodename = ""

    url = string.Template(url).safe_substitute(
        NODE_NAME=nodename, ROS_HOME=rospkg.get_ros_home()
    )

    url = os.path.abspath(url)
    if required and not os.path.exists(url):
        raise Exception("resource not found")

    return url


def get_node_hostname(node_name):
    ID = rospy.get_name()  # my own name
    master = rosgraph.masterapi.Master(ID)

    # Get URI of other node's XMLRPC server.
    uri = master.lookupNode(node_name)

    # Parse the URI to find the hostname of the server.
    x = urlparse(uri)
    if "[" in x.netloc:
        # must be IPv6 according to RFC3986
        if not "]" in x.netloc:
            raise ValueError("invalid IPv6 address")
        start_idx = x.netloc.index("[")
        stop_idx = x.netloc.index("]")
        assert stop_idx > start_idx
        hostname = x.netloc[start_idx + 1 : stop_idx]
    elif ":" in x.netloc:
        hostname = x.netloc[: x.netloc.rindex(":")]
    else:
        hostname = x.netloc

    return hostname
