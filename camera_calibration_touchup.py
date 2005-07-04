import scipy
import scipy.optimize
import cgtypes

# as measured with a ruler
world_measured = [
    [1115, 151, 80],
    [970, 131, 80],
    [984, 251, 80],
    [998, 64, 78],
    [1416, 288, 79],
    [1466, 21, 76],
    [1124, 96, 76],
    [1275, 186, 19],
    ]

# as measured by flydra
world_camera = [
    [1097.9, 107.2, 160.8],
    [950.5, 84.9, 146.5],
    [963.5, 204.1, 139.4],
    [978.5, 24.5, 153.9],
    [1403.0, 243.1, 172.4],
    [1455.7, -32.1, 203.6],
    [1106.2, 55.7, 165.5],
    [1262.8, 131.2, 114.0],
    ]

def get_xform(param_vec):
    rotate=cgtypes.quat(*param_vec[0:4])
    translate=param_vec[4:7]
    scale=param_vec[7:10]

    xform = rotate.toMat4()
    xform.translate( translate )
    xform.scale( scale )
    return xform

def err(param_vec):
    xform = get_xform(param_vec)

    errv = 0

    for iv, ov in zip(world_measured,world_camera):
        iv = cgtypes.vec3(*iv)
        ov = cgtypes.vec3(*ov)
        ovx = xform*ov
        dist = abs(ovx-iv) # distance
        #print ivx, ov, dist
        errv += dist

    print errv, param_vec
    return errv

x0 = (1,0,0,0, 0,0,0, 1,1,1 )

rts = scipy.optimize.fmin_bfgs( err, x0, full_output=0)
xform = get_xform( rts)

#guess of camera coordinates originally used by multicamselfcal
c1 = [1510, 180, 500+322]
c2 = [1140,  260, 450+322]
c3 = [1040, 25, 480+322]
c4 = [490, 160, 390+322]
c5 = [900, 25, 400+322]

#new camera coordinates to use in multicamselfcal
print xform*cgtypes.vec3(*c1)
print xform*cgtypes.vec3(*c2)
print xform*cgtypes.vec3(*c3)
print xform*cgtypes.vec3(*c4)
print xform*cgtypes.vec3(*c5)
