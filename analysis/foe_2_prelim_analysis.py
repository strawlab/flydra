import glob

h5files = glob.glob('*.h5')
logfiles = ['accepted_triggers.txt']
h5files.sort()

pre_frames = 10
post_frames = 100

landed_check_OK = True # relax post_frame requirement iff z<10mm
landed_max_z = 5.0

max_IFI_dist_mm = 20.0
execfile('triggered_prelim.py')
