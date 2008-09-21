MINIMUM_ECCENTRICITY = 1.4 # threshold to fit line

NETWORK_PROTOCOL = 'udp'

timestamp_echo_listener_port = 28992 # on cameras
timestamp_echo_fmt1 = '<d'
timestamp_echo_fmt_diff = '<d'
timestamp_echo_gatherer_port = 28993 # on MainBrain
timestamp_echo_fmt2 = '<dd'

trigger_network_socket_port = 28994 # on MainBrain

# first of potentially many:
min_cam2mainbrain_data_port = 34813 # totally arbitrary, gets communicated
realtime_kalman_port = 28931

emulated_camera_control = 9645
