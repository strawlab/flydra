Estimating orientations with flydra
===================================

.. graphviz::
  strict digraph {
    animals -> onlinePosOri2D;
    animals -> ufmfs;
    onlinePosOri2D -> ufmfs -> imageOri2D -> Ori3D;
    onlinePosOri2D -> ekfPos3D;
    calib -> onlineOri3D;
    calib -> Ori3D;
    calib -> ekfPos3D;
    calib -> imageOri2D;
    ekfPos3D -> Ori3D;
    ekfPos3D -> Ori3D;
    ekfPos3D -> imageOri2D;
    onlinePosOri2D -> onlineOri3D;

    animals [label="experiment"];
    //    onlinePos2D [label="online 2D position estimation"];
    ufmfs [label="saved images (.ufmf)",style=filled];
    imageOri2D [label="image based 2D orientation"];
    Ori3D [label="3D orientation"];
    calib [label="calibration"];
    ekfPos3D [label="EKF-based 3D position tracking"];
    onlineOri3D [label="online 3D orientation"];
    onlinePosOri2D [label="online 2D position and orientation",style=filled];
  } 

* Estimating 2D orientations from images (no description yet)
* :ref:`orientation_ekf_fitter-fusing-2d-orientations-to-3d`
