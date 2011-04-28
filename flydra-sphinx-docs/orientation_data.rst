.. _orientation_data:

Estimating orientations with flydra
===================================

.. graphviz::

  strict digraph {
    animals -> onlinePosOri2D;
    animals -> ufmfs;
    onlinePosOri2D -> ufmfs;
    ufmfs -> IBO [ arrowhead="none"];
    imageOri2D -> OEF [ arrowhead="none"];
    onlinePosOri2D -> retrackedPos2D;
    ufmfs -> retrackedPos2D;
    onlinePosOri2D -> ekfPos3D;
    calib -> onlineOri3D;
    calib -> OEF [ arrowhead="none"];
    calib -> ekfPos3D;
    calib -> IBO [ arrowhead="none"];
    ekfPos3D -> OEF [ arrowhead="none"];
    OEF -> Ori3DHZ;
    Ori3DHZ -> core_analysis [ arrowhead="none"];
    core_analysis -> Ori3D;
    ekfPos3D -> IBO [ arrowhead="none"];
    IBO -> imageOri2D;
    onlinePosOri2D -> onlineOri3D;

    animals [label="experiment"];
    //    onlinePos2D [label="online 2D position estimation"];
    ufmfs [label="saved images (.ufmf)",style=filled];
    imageOri2D [label="image based 2D orientation"];
    IBO [ label = "image_based_orientation", style=filled,color=white ];
    core_analysis [ label = "core_analysis.CachingAnalyzer", style=filled,color=white ];
    OEF [ label = "orientation_ekf_fitter", style=filled,color=white ];
    Ori3DHZ [label="3D orientation, Pluecker coords"];
    Ori3D [label="3D orientation, Cartesian coords"];
    calib [label="calibration"];
    ekfPos3D [label="EKF based 3D position"];
    onlineOri3D [label="online 3D orientation"];
    onlinePosOri2D [label="online 2D position and orientation",style=filled];
  }

Contents:

.. toctree::
  :maxdepth: 2

  orientation_ekf_fitter.rst
  orientation_smoothing.rst

See also :ref:`Data analysis <data_analysis>` (specifically the "Extracting longitudinal body orientation" section).
