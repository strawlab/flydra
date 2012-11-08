import roslib; roslib.load_manifest('rospy')
import rospy

import sys
import errno

class Log:
    to_ros = False
    to_console = True
    force_output = None

    def _write(self, x, f):
        _file = self.force_output if self.force_output != None else f
        if x:
            while 1:
                try:
                    _file.write(x)
                    if not x.endswith('\n'):
                        _file.write('\n')
                    break
                except IOError, err:
                    if err.args[0] == errno.EINTR: # interrupted system call
                        continue
        while 1:
            try:
                _file.flush()
                break
            except IOError, err:
                if err.args[0] == errno.EINTR: 
                    continue

    def info(self, x):
        if self.to_ros:
            #loginfo prints to stdout
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
            #logwarn prints to stderr
            return
        if self.to_console:
            self._write(x, sys.stderr)

    def fatal(self, x):
        if self.to_ros:
            rospy.logfatal(x)
            #logwarn prints to stderr
            return
        if self.to_console:
            self._write(x, sys.stderr)
