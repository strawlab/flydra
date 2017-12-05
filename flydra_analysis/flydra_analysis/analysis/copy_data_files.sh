scp -p -r flygate.caltech.edu:DATA*$DATE*.h5 .
# the following are mounted on brain1 using sshfs
scp -p -r flygate.caltech.edu:cam1/FLYDRA_LARGE_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam2/FLYDRA_LARGE_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam3/FLYDRA_LARGE_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam4/FLYDRA_LARGE_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam5/FLYDRA_LARGE_MOVIES/*$DATE* .

scp -p -r flygate.caltech.edu:cam1/FLYDRA_SMALL_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam2/FLYDRA_SMALL_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam3/FLYDRA_SMALL_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam4/FLYDRA_SMALL_MOVIES/*$DATE* .
scp -p -r flygate.caltech.edu:cam5/FLYDRA_SMALL_MOVIES/*$DATE* .
