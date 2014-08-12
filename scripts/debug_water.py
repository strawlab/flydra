# +                try:
# +                    x_underwater = water.view_points_in_water( self,
# +                                                               cam_id,
# +                                                               underwater_pts,
# +                                                               self.wateri,
# +                                                               distorted=distorted )
# +                except:
# +                    dumper = {'cam_id':cam_id,
# +                              'underwater_pts':underwater_pts,
# +                              'distorted':distorted,
# +                              }
# +                    bad_fname = '/tmp/flydra-water-error.pkl'
# +                    with open(bad_fname,mode='w') as fd:
# +                        pickle.dump(dumper,fd)
# +                    print 'saved data to %s'%bad_fname
# +                    print 'self.wateri',self.wateri
# +                    print 'self',self
# +                    raise

import pickle
import sys

import flydra.water as water
import flydra.reconstruct

pkl_fname = sys.argv[1]

with open(pkl_fname,mode='r') as fd:
    dumper = pickle.load(fd)
print dumper.keys()

cam_id = dumper['cam_id']
underwater_pts = dumper['underwater_pts']
distorted = dumper['distorted']

if 'cal_source' in dumper:
    cal_source = dumper['cal_source']
else:
    cal_source = sys.argv[2]

R = flydra.reconstruct.Reconstructor( cal_source )

x_underwater = water.view_points_in_water( R,
                                           cam_id,
                                           underwater_pts,
                                           R.wateri,
                                           distorted=distorted )
