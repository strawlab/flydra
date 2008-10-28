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
        self._extend_args = []

    def add_parents(self,*parents):
        for parent in parents:
            self.add_parent(parent)

    def add_parent(self,parent):
        assert isinstance(parent,Node)
        self._parents.append( parent )

    def extend_args(self,arg_list):
        self._extend_args.extend( arg_list )

    def make(self,debug=0):
        if self.status == 'up to date':
            return

        # delete non-up-to-date file
        if os.path.exists(self.pathname):
            sys.stderr.write('DELETING %s\n'%self.pathname)
            sys.stderr.write('Press enter to continue, <Ctrl>-C to abort.\n')
            raw_input()
            os.unlink(self.pathname)

        for parent in self._parents:
            parent.make(debug=0)
        cmd_args = self._make()
        cmd_args.extend(self._extend_args)
        if debug>=VERBOSE:
            sys.stdout.write( ' '.join(cmd_args) + '\n' )
        try:
            subprocess.check_call( cmd_args )
        except:
            sys.stderr.write('\n')
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
        if len(instances)==0:
            raise ValueError('Expected an instance of %s, but none found'%klass)
        elif len(instances)>1:
            raise ValueError('Expected one instance of %s, '
                             'but more than one given'%klass)
        return instances[0]

class TwoDeeH5Node(Node):
    pass

class CalibrationNode(Node):
    pass

class FmfNode(Node):
    pass

class UfmfNode(Node):
    def get_pathname_convention_for_associated_mean_fmf(self):
        basename = os.path.splitext(self.pathname)[0]
        return basename + '_mean.fmf'
    def get_associated_mean_fmf(self):
        return FmfNode(
            pathname=self.get_pathname_convention_for_associated_mean_fmf())

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
        cmd_args = ['flydra_analysis_image_based_orientation',
                    '--ufmfs=%s'%ufmf_names,
                    '--h5=%s'%data2d.pathname,
                    '--kalman=%s'%kh5.pathname,
                    '--output-h5=%s'%self.pathname,
                    ]
        return cmd_args

class ViewNode(Node):
    """No output file, just a viewer program"""

class ViewTimeseries2D3DNode(ViewNode):
    def _make(self):
        data2d = self.get_only_instance( TwoDeeH5Node )
        kh5 = self.get_only_instance( KalmanizedH5Node )
        cmd_args = ['flydra_analysis_plot_timeseries_2d_3d',
                    data2d.pathname,
                    '--kalman=%s'%kh5.pathname,
                    ]
        return cmd_args
