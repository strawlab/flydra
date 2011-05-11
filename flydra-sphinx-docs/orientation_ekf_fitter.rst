.. _orientation_ekf_fitter-fusing-2d-orientations-to-3d:

Fusing 2D orientations to 3D
============================

Flydra uses an extended Kalman filter (EKF) and a simple data
association algorithm to fuse 2D orientation data into an a 3D
orientation estimate. The program
:command:`flydra_analysis_orientation_ekf_fitter` is used to perform
this step, and takes, amongst other data, the 2D orientation data
stored in the ``slope`` column of the ``data2d_distorted`` table and
converts it into the ``hz_line*`` columns of the
``kalman_observations`` table. (The directional component of these
Pluecker coordinates should be ignored, as it is meaningless.)

See :ref:`smoothing orientations <orientation_smoothing>` for a
description of the step that chooses orientations (and thus removes
the 180 degree ambiguity in the body orientation estimates).

The following sections detail the algorithm used for the finding of
the hz_line data.

Process model
-------------

We are using a quaternion-based Extended Kalman Filter to track body
orientation and angular rate in 3D.

From (Marins, Yun, Bachmann, McGhee, and Zyda, 2001) we have state
:math:`\boldsymbol{{\mathrm x}}=x_1,x_2,x_3,x_4,x_5,x_6,x_7` where
:math:`x_1,x_2,x_3` are angular rates :math:`p,q,r` and
:math:`x_4,x_5,x_6,x_y` are quaternion components :math:`a,b,c,d`
(with the scalar component being :math:`d`).

The temporal derivative of :math:`\boldsymbol{{\mathrm x}}` is
:math:`\dot{\boldsymbol{{\mathrm x}}}=f(\boldsymbol{{\mathrm x}})` and
is defined as:

.. math::

  \left(\begin{smallmatrix}- \frac{x_{1}}{\tau_{rx}} & - \frac{x_{2}}{\tau_{ry}} & - \frac{x_{3}}{\tau_{rz}} & \frac{x_{1} x_{7} + x_{3} x_{5} - x_{2} x_{6}}{2 \sqrt{{x_{4}}^{2} + {x_{5}}^{2} + {x_{6}}^{2} + {x_{7}}^{2}}} & \frac{x_{1} x_{6} + x_{2} x_{7} - x_{3} x_{4}}{2 \sqrt{{x_{4}}^{2} + {x_{5}}^{2} + {x_{6}}^{2} + {x_{7}}^{2}}} & \frac{x_{2} x_{4} + x_{3} x_{7} - x_{1} x_{5}}{2 \sqrt{{x_{4}}^{2} + {x_{5}}^{2} + {x_{6}}^{2} + {x_{7}}^{2}}} & \frac{x_{3} x_{6} - x_{1} x_{4} - x_{2} x_{5}}{2 \sqrt{{x_{4}}^{2} + {x_{5}}^{2} + {x_{6}}^{2} + {x_{7}}^{2}}}\end{smallmatrix}\right)

The process update equation (for :math:`\boldsymbol{{\mathrm x}}_t \vert \boldsymbol{{\mathrm x}}_{t-1}`) is:

.. math::

  \boldsymbol{{\mathrm x}}_{t+1} = \boldsymbol{{\mathrm x}}_t + 
                                   f(\boldsymbol{{\mathrm x}}_t) dt + 
                                   \boldsymbol{{\mathrm w}}_t

Where :math:`\boldsymbol{{\mathrm w}}_t` is the noise term with
covariance :math:`Q` and :math:`dt` is the time step.



Observation model
-----------------

The goal is to model how the target orientation given by quaternion
:math:`q=a i+b j + c k + d` results in a line on the image, and
finally, the angle of that line on the image. We also need to know the
target 3D location, the vector :math:`A`, and the camera matrix
:math:`P`. Thus, the goal is to define the function
:math:`G(q,A,P)=\theta`.

Quaternion :math:`q` may be used to rotate the vector :math:`u` using
the matrix R:

.. math::

  R = \left(\begin{smallmatrix}{a}^{2} + {d}^{2} - {b}^{2} - {c}^{2} & - 2 c d + 2 a b & 2 a c + 2 b d\\2 a b + 2 c d & {b}^{2} + {d}^{2} - {a}^{2} - {c}^{2} & - 2 a d + 2 b c\\- 2 b d + 2 a c & 2 a d + 2 b c & {c}^{2} + {d}^{2} - {a}^{2} - {b}^{2}\end{smallmatrix}\right)

Thus, for :math:`u=(1,0,0)` the default, non-rotated orientation, we
find :math:`U=Ru`, the orientation estimate.

.. math::

  U=Ru = \left(\begin{smallmatrix}{a}^{2} + {d}^{2} - {b}^{2} - {c}^{2}\\2 a b + 2 c d\\- 2 b d + 2 a c\end{smallmatrix}\right)

Now, considering a point passing through :math:`A` with orientation
given by :math:`U`, we define a second point :math:`B=A+U`.

Given the camera matrix :math:`P`:

.. math::

  P = \left(\begin{smallmatrix}P_{00} & P_{01} & P_{02} & P_{03}\\P_{10} & P_{11} & P_{12} & P_{13}\\P_{20} & P_{21} & P_{22} & P_{23}\end{smallmatrix}\right)

The image of point :math:`A` is :math:`PA`. Thus the vec on the image is :math:`PB-PA`.

.. math::

  G(q,A,P) = \theta = \operatorname{atan}\left(\frac{\frac{P_{13} + Ax P_{10} + Ay P_{11} + Az P_{12} - 2 P_{12} b d + 2 P_{11} a b + 2 P_{11} c d + 2 P_{12} a c + P_{10} {a}^{2} + P_{10} {d}^{2} - P_{10} {b}^{2} - P_{10} {c}^{2}}{P_{23} + Ax P_{20} + Ay P_{21} + Az P_{22} - 2 P_{22} b d + 2 P_{21} a b + 2 P_{21} c d + 2 P_{22} a c + P_{20} {a}^{2} + P_{20} {d}^{2} - P_{20} {b}^{2} - P_{20} {c}^{2}} - \frac{P_{13} + Ax P_{10} + Ay P_{11} + Az P_{12}}{P_{23} + Ax P_{20} + Ay P_{21} + Az P_{22}}}{\frac{P_{03} + Ax P_{00} + Ay P_{01} + Az P_{02} - 2 P_{02} b d + 2 P_{01} a b + 2 P_{01} c d + 2 P_{02} a c + P_{00} {a}^{2} + P_{00} {d}^{2} - P_{00} {b}^{2} - P_{00} {c}^{2}}{P_{23} + Ax P_{20} + Ay P_{21} + Az P_{22} - 2 P_{22} b d + 2 P_{21} a b + 2 P_{21} c d + 2 P_{22} a c + P_{20} {a}^{2} + P_{20} {d}^{2} - P_{20} {b}^{2} - P_{20} {c}^{2}} - \frac{P_{03} + Ax P_{00} + Ay P_{01} + Az P_{02}}{P_{23} + Ax P_{20} + Ay P_{21} + Az P_{22}}}\right)

Now, we need to shift coordinate system such that angles will be small
and thus reasonably approximated by normal distributions. We thus take
an expected orientation quaternion :math:`q^\ast` and find the
expected image angle for that :math:`\theta^\ast`:

.. math::

  \Phi(q^\ast,A,P) = \theta^\ast

We define our new observation model in this coordinate system:

.. math::

  H(q,q^\ast,A,P) = G(q,A,P) - \Phi(q^\ast,A,P) = \theta - \theta^\ast

Of course, if the original observation was :math:`y`, the new
observation :math:`z` must also be placed in this coordinate system.

.. math::
  
  z_t = y_t - \theta^\ast

The EKF prior estimate of orientation is used as the expected
orientation :math:`q^\ast`, although is possible to use other values
for expected orientation.

Data association
----------------

The data association follows a very simple rule. An observation
:math:`z_t` is used if and only if this value is close to the expected
value. Due to the definition of :math:`z_t` above, this is equivalent
to saying only small absolute values of :math:`z_t` are associated
with the target. This gating is established by the
``--gate-angle-threshold-degrees`` parameter to
:command:`flydra_analysis_orientation_ekf_fitter`. ``--gate-angle-threshold-degrees``
is defined between 0 and 180. The basic idea is that the program has a
prior estimate of orientation and angular velocity from past frames,
and any new 2D orientation is accepted or not (gated) based on whether
the acceptance makes sense -- whether it's close to the predicted
value. So a value of zero means reject everything and 180 means accept
everything. 10 means that you believe your prior estimates and only
accept close observations, where as 170 means you think the prior is
less reliable than the observation. (IIRC, the presence or absence of
the green line in the videos indicates whether the 2D orientation was
gated in or out, respectively.)

``--area-threshold-for-orientation`` lets you discard a point if the
area of the 2D detection is too low. Many spurious detections often
have really low area, so this is a good way to get rid of
them. However, the default of this value is zero, so I think when I
wrote the program I found it to be unnecessary.
