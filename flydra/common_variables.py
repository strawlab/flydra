MINIMUM_ECCENTRICITY = 1.4 # threshold to fit line
ACCEPTABLE_DISTANCE_PIXELS = 10.0
#REALTIME_UDP = False # use UDP or TCP to send realtime data
REALTIME_UDP = True # use UDP or TCP to send realtime data

timestamp_echo_listener_port = 28992 # on cameras
timestamp_echo_fmt1 = '<d'
timestamp_echo_fmt_diff = '<d'
timestamp_echo_gatherer_port = 28993 # on MainBrain
timestamp_echo_fmt2 = '<dd'

# first of potentially many:
min_cam2mainbrain_data_port = 34813 # totally arbitrary, get communicated
