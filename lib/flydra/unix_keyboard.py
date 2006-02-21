import sys, tty, termios, threading, time, select, Queue, atexit

__all__ = ['getch','kbhit']

class _MyClass:
    def __init__(self):
        self.chqueue = Queue.Queue()
        self.thread = threading.Thread( target=self._run_func )
        self.thread.setDaemon(True)
        self.thread.start()
        
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
        
        atexit.register(self._atexit) # revert console on exit

    def _run_func( self ):
        fd = sys.stdin
        fdr = fd.read
        fl = [fd]
        nl = []
        s = select.select
        qp = self.chqueue.put
        while 1:
            rl=s(fl,nl,nl, 0.01)[0]
            if len(rl):
                ch = fdr(1)
                qp(ch)

    def _atexit(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            
    def getch(self):
        """blocks and returns next character"""
        return self.chqueue.get(True,None) # block, no timeout

    def kbhit(self):
        """returns True is a character is waiting"""
        return not self.chqueue.empty()

_myc = _MyClass()
kbhit = _myc.kbhit
getch = _myc.getch

if __name__=='__main__':
    while 1:
        time.sleep(0.1)
        if kbhit():
            print 'hit:',getch()
        else:
            print 'nothing'
        
