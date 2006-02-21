import scipy
import scipy.optimize
import cgtypes

# as measured with a ruler
world_measured = [
    [517, 43, 74],
    [509, 174, 76],
    [585, 283, 69],
    [699, 256, 69],
    [703, 39, 70],
    [1074, 39, 71],
    [1024, 230, 73],
    [809, 236, 17],
    [760, 79, 15],
    [598, 178, 9],
    ]

# as measured by flydra
world_camera = [
    [508, -1, 56.4],
    [494.9, 109.8, 40.4],
    [560.1, 213, 14.4],
    [668.7, 199.4, 22.2],
    [679, -3.9, 56.0],
    [1012.8, 14.6, 64],
    [972.5, 188.5, 33.4],
    [772.4, 164.9, -16.9],
    [730.6, 22.1, 1.7],
    [577, 100.7, -21.5],
    ]
    
##    [805.5, 214.7, 181.5],
##    [509.2, 305.4, -11.4],
##    [527.5, 134.2, -8.6],
##    [646.3, 43.2, -10.6],
##    [1099.7, 358.3, 13.0],
##    [947.9, 146.4, 9.3],
##    [837.8, 51.9, -0.4],
##    [775.4, 212.2, -3.1],
##    [629.9, 374.1, -13.1],
##    [650.3, 206.7, -4.8],
##    ]

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

#guess of camera coordinates originally used by multicamselfcal (flydra.m)

c1 = [1710, 375, 820]
c2= [1110, 316, 800]
c3 = [1110, 100, 466]
c4 = [-380, 290, 875]
c5 = [440, 15, 680]

#new camera coordinates to use in multicamselfcal
print xform*cgtypes.vec3(*c1)
print xform*cgtypes.vec3(*c2)
print xform*cgtypes.vec3(*c3)
print xform*cgtypes.vec3(*c4)
print xform*cgtypes.vec3(*c5)
