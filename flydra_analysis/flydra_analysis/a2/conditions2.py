# this is some kind of order
condition_names = ["analysis treatments"]

files = {
    "analysis treatments": [
        "DATA20070202_190006.h5",
        "DATA20070202_190006.kalmanized.h5",  # recovered 2D data used (re-kalmanized, but not using new calibration)
        #'DATA20070202_190006.copy.h5', # backup copy of original file (DON'T USE), same as original
        "DATA20070202_190006.copy.kalmanized.h5",  # recalibrated from same day's data, recovered 2D data used
    ],
}
