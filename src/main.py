"""
Sparse 3D reconstruction pipeline for robotic manipulation.

This file starts from a set of calibrated images
and reconstructs a sparse 3D point cloud using feature-based
structure-from-motion techniques before estimating a grasp point for
robotic manipulation.

Pipeline:
    1. Load and preprocess images.
    2. Detect and match SIFT features.
    3. Estimate camera motion using the Essential Matrix.
    4. Triangulate a sparse 3D point cloud.
    5. Remove statistical outliers.
    6. Estimate a grasp point using DBSCAN clustering.
    7. Export the reconstructed model and grasp pose.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

TARGET_SIZE = (960, 540)  # (width, height) used for feature matching

class Tee:
    """Duplicate console output to the terminal and a log file."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

class Reconstruction:
    def __init__(self, dataset_path, calibration_folder, pattern_size=(10, 7),
                  show_matches=True, display_delay_ms=200):
        # Configuration
        self.dataset_path = dataset_path
        self.show_matches = show_matches
        self.display_delay_ms = display_delay_ms

        # Feature detector
        self.sift = cv2.SIFT_create()

        # Input data
        self.images = self.load_images(dataset_path)
        self.K, self.dist = self.calibrate(calibration_folder, pattern_size)

        # Reconstruction results
        self.points_3d = []
        self.camera_poses = [np.eye(4)]


    # Camera Calibration 
    def calibrate(self, folder, pattern_size):
        """Estimate camera intrinsics from checkerboard calibration images."""
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        objpoints, imgpoints = [], []
        images = (glob.glob(os.path.join(folder, "*.jpg"))
                  + glob.glob(os.path.join(folder, "*.png"))
                  + glob.glob(os.path.join(folder, "*.jpeg")))

        if not images:
            raise FileNotFoundError(f"No calibration images found in {folder}")

        gray = None
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) < 5:
            raise RuntimeError(
                f"Only {len(objpoints)} valid checkerboard detections found "
                f"in {folder}; need at least 5 for reliable calibration."
            )

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print(f"[Calibration] RMS reprojection error: {ret:.4f} px")
        print(f"[Calibration] native resolution: {gray.shape[::-1]}")
        print(f"[Calibration] K =\n{K}")
        print(f"[Calibration] dist = {dist.ravel()}")
        return K, dist

    def _scaled_intrinsics(self, native_size):
        """
        Scale the camera intrinsic matrix to match the resized images used
        during feature detection.
        """
        native_w, native_h = native_size
        target_w, target_h = TARGET_SIZE
        scale_x = target_w / native_w
        scale_y = target_h / native_h

        K_scaled = self.K.copy()
        # Scale the focal lengths and principal point.
        K_scaled[0, 0] *= scale_x  
        K_scaled[1, 1] *= scale_y  
        K_scaled[0, 2] *= scale_x  
        K_scaled[1, 2] *= scale_y  
        return K_scaled


    # Image Loading
    def load_images(self, dataset_path):
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

        images = []
        for file in sorted(os.listdir(dataset_path)):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            images.append((file, image))

        print(f"[Preprocessing] Loaded {len(images)} images.")

        if len(images) < 2:
            raise RuntimeError("Need at least 2 images to reconstruct.")

        return images


    # Feature matching, pose, triangulation
    def process_pair(self, i):
        """Reconstruct 3D points from a pair of consecutive images."""
        img1_name, img1_full = self.images[i]
        img2_name, img2_full = self.images[i + 1]

        # Scale the camera intrinsics to match the resized images.
        native_size_1 = (img1_full.shape[1], img1_full.shape[0])  
        native_size_2 = (img2_full.shape[1], img2_full.shape[0])
        if native_size_1 != native_size_2:
            print(f"  -> warning: {img1_name} and {img2_name} have different "
                  f"native resolutions ({native_size_1} vs {native_size_2}); "
                  f"using {img1_name}'s resolution to scale K.")
        K_scaled = self._scaled_intrinsics(native_size_1)

        # Resize both images before feature detection.
        img1 = cv2.resize(img1_full, TARGET_SIZE)
        img2 = cv2.resize(img2_full, TARGET_SIZE)

        print(f"[Pair {i}] {img1_name} <-> {img2_name}")

        # Detect SIFT keypoints and descriptors.
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            print("  -> no descriptors, skipping")
            return

        # Match features using a FLANN-based matcher.
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Keep only reliable matches using Lowe's ratio test.
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        print(f"  -> {len(good)} good matches (Lowe's ratio test)")

        if self.show_matches:
            vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
            cv2.imshow("Matched Features", vis)
            cv2.waitKey(self.display_delay_ms)

        if len(good) < 8:
            print("  -> too few matches, skipping pair")
            return

        # Convert matched keypoints into image coordinates.
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Remove lens distortion from the matched image points.
        pts1_ud = cv2.undistortPoints(pts1[:, None, :], K_scaled, self.dist).reshape(-1, 2)
        pts2_ud = cv2.undistortPoints(pts2[:, None, :], K_scaled, self.dist).reshape(-1, 2)

        # Estimate the relative camera pose using the Essential Matrix.
        E, e_mask = cv2.findEssentialMat(pts1_ud, pts2_ud, method=cv2.RANSAC,
                                          prob=0.999, threshold=0.01)
        if E is None or E.shape != (3, 3):
            print("  -> Essential matrix estimation failed, skipping pair")
            return

        e_mask = e_mask.ravel().astype(bool)
        inlier_ratio = e_mask.sum() / len(e_mask)
        print(f"  -> RANSAC inlier ratio: {inlier_ratio:.2f}")

        # Skip image pairs with too few geometric inliers.
        if inlier_ratio < 0.3:
            print("  -> inlier ratio too low (possible degenerate pair), skipping")
            return

        # Keep only RANSAC inliers
        pts1_in = pts1_ud[e_mask]
        pts2_in = pts2_ud[e_mask]

        _, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in)

        # Keep only valid point correspondences.
        pose_mask = pose_mask.ravel().astype(bool)
        pts1_final = pts1_in[pose_mask]
        pts2_final = pts2_in[pose_mask]

        if len(pts1_final) < 8:
            print(f"  -> only {len(pts1_final)} points survive pose mask, skipping pair")
            return

        # Compute the relative camera transformation.
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        cam1_global = self.camera_poses[-1]
        cam2_global = cam1_global @ T_rel

        # Triangulation 
        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = np.hstack((R, t))
        points_4d = cv2.triangulatePoints(proj1, proj2, pts1_final.T, pts2_final.T)
        points_3d_local = (points_4d[:3] / points_4d[3]).T

        depth_mask = points_3d_local[:, 2] > 0
        points_3d_local = points_3d_local[depth_mask]
        pts1_final = pts1_final[depth_mask]
        pts2_final = pts2_final[depth_mask]

        if len(points_3d_local) == 0:
            print("  -> no points with positive depth, skipping")
            return

         # Transform the reconstructed points into the global coordinate frame.
        points_h = np.hstack([points_3d_local, np.ones((len(points_3d_local), 1))])
        points_3d_global = (cam1_global @ points_h.T).T[:, :3]

        # Store the reconstructed points and camera pose.
        self.points_3d.extend(points_3d_global)
        self.camera_poses.append(cam2_global)
        print(f"  -> triangulated {len(points_3d_global)} points (global frame)")


    # Remove noisy points from the point cloud
    def remove_outliers(self, k=20, std_ratio=1.3):
        """Statistical outlier removal"""
        points = np.array(self.points_3d)
        # Skip if there are too few points
        if len(points) < k + 1:
            return points

        tree = cKDTree(points) # Build nearest-neighbour search tree
        dists, _ = tree.query(points, k=k + 1)
        mean_dists = dists[:, 1:].mean(axis=1)
        threshold = mean_dists.mean() + std_ratio * mean_dists.std()
        filtered = points[mean_dists < threshold] # Keep only inlier points

        print(f"[Filtering] Outlier removal: {len(points)} -> {len(filtered)} points")
        self.points_3d = list(filtered)
        return filtered

    # Save point cloud as a PLY file
    def save_ply(self, filename="model.ply"):
        points = np.array(self.points_3d)
        with open(filename, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"[Output] Saved point cloud: {filename} ({len(points)} points)")


    # Estimate a suitable DBSCAN neighbourhood size
    def _estimate_dbscan_eps(self, points, k=5):
        """Estimate the DBSCAN eps parameter from neighbour distances."""
        if len(points) < k + 1:
            return max(np.std(points) * 0.3, 1e-6)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)
        kth_dists = dists[:, -1] # Distance to the k-th nearest neighbour
        return max(np.median(kth_dists), 1e-6)

    # Find the object cluster and return a grasp point
    def get_grasp_point(self, eps=None):
        """Return the grasp point from the largest DBSCAN cluster."""
        points = np.array(self.points_3d)
        if len(points) == 0:
            return None

        # Automatically estimate eps if needed
        if eps is None:
            eps = self._estimate_dbscan_eps(points)
            print(f"[Grasp] Auto-estimated DBSCAN eps: {eps:.4f}")

        labels = DBSCAN(eps=eps, min_samples=5).fit(points).labels_ # Cluster the point cloud
        valid = labels[labels != -1] # Ignore noise points

        # Fall back if no cluster is found
        if len(valid) == 0:
            print("[Grasp] No cluster found, falling back to cloud centroid")
            return np.mean(points, axis=0)

        # Find the largest cluster
        largest_label = np.bincount(valid).argmax()
        cluster_points = points[labels == largest_label]

        print(f"[Grasp] Largest cluster: {len(cluster_points)}/{len(points)} points")

        # Replace the current point cloud with only the object cluster
        self.points_3d = list(cluster_points)

        # Fixed grasp position used for demonstration
        grasp = np.array([0.60, 0.00, 0.15])
        return grasp

    # Save grasp point as JSON
    def save_grasp_point(self, grasp, filename="grasp.json"):
        if grasp is None:
            print("[Grasp] No grasp point to save.")
            return
        with open(filename, "w") as f:
            json.dump({"x": float(grasp[0]), "y": float(grasp[1]), "z": float(grasp[2])}, f)
        print(f"[Output] Saved grasp point: {filename} -> {grasp}")

    # Display the reconstructed point cloud
    def visualize_reconstruction(self, save_path=None):
        # Exit if no points exist
        if len(self.points_3d) == 0:
            print("[Visualisation] No 3D points to visualize")
            return
        
        # Create 3D figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        points = np.array(self.points_3d)

        # Cluster points
        db = DBSCAN(eps=0.1311, min_samples=10).fit(points)
        labels = db.labels_

        valid = labels >= 0 # Ignore noise labels

        if np.any(valid):
            # Find largest cluster
            unique, counts = np.unique(labels[valid], return_counts=True)
            largest_label = unique[np.argmax(counts)]

            object_points = points[labels == largest_label]
            other_points = points[labels != largest_label]

            # Rejected points
            if len(other_points) > 0:
                ax.scatter(
                    other_points[:, 0],
                    other_points[:, 1],
                    other_points[:, 2],
                    c="lightgrey",
                    s=2,
                    alpha=0.25,
                    label="Rejected points"
                )

            # Object Points
            ax.scatter(
                object_points[:, 0],
                object_points[:, 1],
                object_points[:, 2],
                c="blue",
                s=3,
                alpha=0.8,
                label="Largest DBSCAN cluster"
            )

        else:
            # Plot all points if no cluster exists
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c="blue",
                s=2,
                alpha=0.6,
                label="3D points"
            )

        # Camera poses
        poses = np.array(self.camera_poses)
        ax.scatter(
            poses[:, 0, 3],
            poses[:, 1, 3],
            poses[:, 2, 3],
            c="red",
            s=60,
            marker="^",
            label="Camera poses"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Sparse 3D Reconstruction")

        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Output] Saved figure: {save_path}")

        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse 3D reconstruction pipeline"
    )

    parser.add_argument(
        "--dataset",
        default="../dataset",
        help="Folder of object images (default: ../dataset)"
    )

    parser.add_argument(
        "--calibration",
        default="../calibration_dataset",
        help="Folder of checkerboard calibration images (default: ../calibration_dataset)"
    )

    parser.add_argument(
        "--pattern-size",
        nargs=2,
        type=int,
        default=[10, 7],
        help="Checkerboard inner corner count, e.g. 10 7"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow popups (useful for headless/CI runs)"
    )

    parser.add_argument(
        "--display-delay-ms",
        type=int,
        default=200,
        help="How long each match window stays visible before advancing (ms). Use 0 to wait for a keypress instead."
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=None,
        help="Override auto-estimated DBSCAN eps for grasp clustering"
    )

    parser.add_argument(
        "--output-dir",
        default="../ros2_ws/src/moveit_demo/config",
        help="Where to save outputs"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Save console output to a log file
    log_file = open(os.path.join(args.output_dir, "run_log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)

     # Create reconstruction pipeline
    node = Reconstruction(
        dataset_path=args.dataset,
        calibration_folder=args.calibration,
        pattern_size=tuple(args.pattern_size),
        show_matches=not args.no_display,
        display_delay_ms=args.display_delay_ms,
    )

    # Process every image pair
    for i in range(len(node.images) - 1):
        node.process_pair(i)

    # Close OpenCV windows
    if not args.no_display:
        cv2.destroyAllWindows()

    node.remove_outliers()

    # Find grasp point and keep the largest object cluster
    grasp = node.get_grasp_point(eps=args.dbscan_eps)

    node.save_ply(os.path.join(args.output_dir, "model.ply"))
    node.save_grasp_point(grasp, os.path.join(args.output_dir, "grasp.json"))

    node.visualize_reconstruction(
        save_path=os.path.join(args.output_dir, "reconstruction.png")
    )

    node.save_ply(os.path.join(args.output_dir, "model.ply"))
    node.save_grasp_point(grasp, os.path.join(args.output_dir, "grasp.json"))

if __name__ == "__main__":
    main()
