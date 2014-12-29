MINIMUM_ECCENTRICITY = 1.4 # threshold to fit line

# epsilon for numerical precision when computing quartic roots in
# refractive boundary code
WATER_ROOTS_EPS = 1e-5

timestamp_echo_listener_port = 9992 # on cameras
timestamp_echo_fmt1 = '<d'
timestamp_echo_fmt_diff = '<d'
timestamp_echo_gatherer_port = 9993 # on MainBrain
timestamp_echo_fmt2 = '<dd'

emulated_camera_control = 9645

sync_duration = 2.0 # seconds to pause triggering to cause synchronization
near_inf = 9.999999e20
