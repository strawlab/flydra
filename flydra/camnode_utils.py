#emacs, this is -*-Python-*- mode
from __future__ import division
from __future__ import with_statement

import contextlib

import threading, Queue

class ChainLink(object):
    """essentially a linked list of threads"""
    def __init__(self):
        self._queue = Queue.Queue()

        self._lock = threading.Lock()
        # start: vars access controlled by self._lock
        self._next = None
        #   end: vars access controlled by self._lock

    def fire(self, buf):
        """fire a listener in new thread. Threadsafe"""
        self._queue.put( buf )

    def append_link(self, chain ):
        if not isinstance(chain,ChainLink):
            raise ValueError("%s is not instance of ChainLink"%(str(chain),))
        else:
            print "%s is instance of ChainLink"%(str(chain),)
        with self._lock:
            if self._next is None:
                self._next = chain
                return
            else:
                next = self._next
        next.append_link( chain )

    def get_buf(self):
        """called from client thread to get a buffer"""
        return self._queue.get()

    def end_buf(self,buf):
        """called from client thread to release a buffer"""
        with self._lock:
            next = self._next

        if next is not None:
            next.fire(buf)
        else:
            pool = buf.get_pool()
            pool.return_buffer( buf )

@contextlib.contextmanager
def use_buffer_from_chain(link):
    """manage access to the buffer"""
    buf = link.get_buf()
    try:
        yield buf
    finally:
        link.end_buf(buf)
