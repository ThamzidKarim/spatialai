# 3D Reconstruction for Robotic Manipulation

A Python/OpenCV Structure-from-Motion (SfM) pipeline that reconstructs a sparse 3D point cloud from phone images, filters noise, segments the target object, and integrates it into a MoveItPy planning scene.

**Technologies:** Python, OpenCV, SIFT, FLANN, RANSAC, triangulation, DBSCAN, ROS 2, MoveItPy.

## Results

- Processed **24 images** at 960 × 540 resolution.
- Reconstructed **11,377 3D points**, reduced to **10,931** after outlier filtering (3.92% removed).
- Identified a largest DBSCAN cluster of **1,119 points** for the exported object model.
- Achieved a camera calibration RMS reprojection error of **1.7312 pixels**.
- Exported a 3D point cloud (`model.ply`) and grasp position (`grasp.json`).
- Integrated the reconstructed object into MoveItPy for collision-aware pick-and-place planning.

![Sparse 3D reconstruction](reconstruction4.png)

## Pipeline

1. Calibrate the camera using checkerboard images.
2. Detect and match SIFT features using FLANN and Lowe's ratio test.
3. Estimate camera poses using the essential matrix and RANSAC; triangulate matched features.
4. Remove outliers and segment the main point cluster using DBSCAN.
5. Export the reconstruction for robotic motion planning.

## Run

```bash
python3 main.py
```

Requires the input image and calibration datasets at the paths expected by the code.

The robotics component additionally requires a configured ROS 2, MoveIt 2 and UR5e environment.

## Limitations

The reconstruction is sparse and depends on reliable feature matching. The largest DBSCAN cluster is assumed to be the target object. The grasp position is predefined, and collision-object dimensions are fixed rather than estimated from the reconstruction.

*Results are from one run. Ground-truth 3D accuracy and segmentation precision/recall were not evaluated.*
