"""FreshScan: detecting replayed LiDAR frames with ego-motion consistency.

Modules
-------
kitti_loader : read KITTI raw synced drives (Velodyne frames, OXTS motion, LiDAR timestamps)
replay       : replay injection (sigma(t) = t - R inside random windows) with frame labels
registration : C1 preparation (crop + voxel) and C2 LiDAR ego-motion by point-to-point ICP
features     : C3 consistency checks (motion residuals and seam features)
detector     : C5 decision (benign-normalized max score, m-of-n alarm, trust) and B1 timestamp check
"""

__version__ = "1.0.0"
