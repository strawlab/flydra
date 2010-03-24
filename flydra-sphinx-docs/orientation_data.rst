.. _orientation_data:

Estimating orientations with flydra
===================================

.. graphviz::
  strict digraph {
    animals -> onlinePosOri2D;
    animals -> ufmfs;
    onlinePosOri2D -> ufmfs -> imageOri2D -> Ori3D;
    onlinePosOri2D -> retrackedPos2D;
    ufmfs -> retrackedPos2D;
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
    ekfPos3D [label="EKF based 3D position"];
    onlineOri3D [label="online 3D orientation"];
    onlinePosOri2D [label="online 2D position and orientation",style=filled];
  } 

Contents:

.. toctree::
  :maxdepth: 2

  orientation_ekf_fitter.rst

* Estimating 2D orientations from images (no description yet)

