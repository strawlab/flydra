from __future__ import print_function
from __future__ import absolute_import
from .PQmath import orientation_to_quat, quat_to_orient
import numpy
import cgtypes

q = orientation_to_quat((1, 0, 0))
angles = numpy.linspace(0, 2 * numpy.pi, 10)


def rotate_quat(quat, rotation_quat):
    res = quat * rotation_quat * quat * quat.inverse()
    return res


for angle in angles:
    rotation_quat = cgtypes.quat().fromAngleAxis(angle, (0, 0, 1))
    q2 = rotate_quat(q, rotation_quat)
    o = quat_to_orient(q2)
    print(o)

print("-=" * 20)
# 1 rev per second
angular_velocity = cgtypes.vec3((0, 0, 2 * numpy.pi))

times = numpy.linspace(0, 1.0, 10)
for t in times:
    ang_change = float(t) * angular_velocity
    angle = abs(ang_change)
    if angle == 0.0:
        axis = cgtypes.vec3((1, 0, 0))
    else:
        axis = ang_change.normalize()
    # print 'axis',axis
    # print 'angle',angle
    rotation_quat = cgtypes.quat().fromAngleAxis(angle, axis)
    q2 = rotate_quat(q, rotation_quat)
    o = quat_to_orient(q2)
    print(o)

print("-=" * 20)

# cumulative addition of angle given angular velocity

dt = float(times[1] - times[0])
t = -dt
q2 = q
for i in range(len(times)):
    t += dt
    o = quat_to_orient(q2)

    ang_change = dt * angular_velocity
    angle = abs(ang_change)
    if angle == 0.0:
        axis = cgtypes.vec3((1, 0, 0))
    else:
        axis = ang_change.normalize()
    rotation_quat = cgtypes.quat().fromAngleAxis(angle, axis)
    q2 = rotate_quat(q2, rotation_quat)
    print(o)
