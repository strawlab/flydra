import os, subprocess, sys
VERBOSE=1
# Directed Acyclic Graphs to analyze data
class Node(object):
    def __init__(self, pathname=None, status='unknown', parents=None):
        if parents is None:
            self._parents = []
        else:
            self._parents = parents
        self.pathname=pathname
        self.status=status

    def add_parents(self,*parents):
        for parent in parents:
            self.add_parent(parent)

    def add_parent(self,parent):
        assert isinstance(parent,Node)
        self._parents.append( parent )

    def make(self,debug=0):
        if self.status == 'up to date':
            return

        for parent in self._parents:
            parent.make(debug=0)
        cmd_args = self._make()
        if debug>=VERBOSE:
            sys.stdout.write( ' '.join(cmd_args) + '\n' )
        try:
            subprocess.check_call( cmd_args )
        except:
            sys.exit(1)
        self.status='up to date'

    def _make(self):
        # The parents are guaranteed to be made when this is called
        # and status is not 'up to date'.
        raise NotImplementedError('abstract base class method called')

    def get_parent_instances( self, klass ):
        results = [ i for i in self._parents if isinstance(i,klass) ]
        return results

    def get_only_instance(self,klass):
        instances = self.get_parent_instances(klass)
        assert len(instances)==1
        return instances[0]

class TwoDeeH5Node(Node):
    pass

class CalibrationNode(Node):
    pass

class UfmfNode(Node):
    pass

class KalmanizedH5Node(Node):
    def _make(self):
        cal = self.get_only_instance( CalibrationNode )
        data2d = self.get_only_instance( TwoDeeH5Node )

        if os.path.exists(self.pathname):
            os.unlink(self.pathname)

        cmd_args = ['flydra_kalmanize',
                    data2d.pathname,
                    '--reconstructor=%s'%cal.pathname,
                    '--dest-file=%s'%self.pathname,
                    ]
        return cmd_args

class ImageBasedOrientationNode(TwoDeeH5Node):
    def _make(self):
        ufmfs = self.get_parent_instances( UfmfNode )
        data2d = self.get_only_instance( TwoDeeH5Node )
        kh5 = self.get_only_instance( KalmanizedH5Node )
        ufmf_names = os.pathsep.join([ufmf.pathname for ufmf in ufmfs])
        cmd_args = ['python',
                    '/home/astraw/src/kookaburra.git/flydra/flydra/a2/image_based_orientation.py',
                    '--ufmfs=%s'%ufmf_names,
                    '--h5=%s'%data2d.pathname,
                    '--kalman=%s'%kh5.pathname,
                    ]
        return cmd_args
