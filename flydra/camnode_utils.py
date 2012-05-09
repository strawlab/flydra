#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import contextlib

import threading, Queue
import rospy

class ChainLink(object):
    """Essentially a threadsafe linked list of image queues."""
    def __init__(self):
        self._queue = Queue.Queue()
        self._lock = threading.Lock()
        # start: vars access controlled by self._lock
        self._next = None
        #   end: vars access controlled by self._lock

    def append_chain(self, chain):
        if not isinstance(chain, ChainLink):
            raise ValueError("%s is not instance of ChainLink"%(str(chain),))

        with self._lock:
            if self._next is None:
                self._next = chain
                return
            else:
                next = self._next
        next.append_chain( chain )

    def put(self, image):
        """put an image into queue."""
        with self._lock:
            #rospy.logwarn('put(%s)' % image)
            self._queue.put(image)

    def get(self, blocking=True):
        """Called from client thread to get a image."""
        image = self._queue.get(blocking)
        #rospy.logwarn('get(%s)' % image)
        return image
    

    def release(self, image):
        """Called from client thread to release an image."""
        #rospy.logwarn('release(%s)' % image)
        with self._lock:
            next = self._next

        if next is not None:
            next.put(image)           # Kick the image down to the end chainlink.
        else:
            pool = image.get_pool()   # If we're the end, then release the image into its pool.
            pool.release(image)

@contextlib.contextmanager
def use_buffer_from_chain(chain, blocking=True):
    """manage access to the buffer"""
    image = chain.get(blocking=blocking)
    try:
        yield image
    finally:
        chain.release(image)
