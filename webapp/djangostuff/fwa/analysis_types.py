class AnalysisType(object):
    pass

class EKF_based_3D_position( AnalysisType ):
    name = 'EKF-based 3D position'
    short_description = 'convert 2D data and calibration into 3D position data'
    parent_node_types = ['2d position', 'calibration']

class_names = ['EKF_based_3D_position']
