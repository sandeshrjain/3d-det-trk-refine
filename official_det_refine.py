# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:57:23 2023
@author: Sandesh Jain
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import KMeans
from collections import Counter

# Define mapping dictionaries
kitti_label_dict = {
    0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle",
    16: "on-rails", 18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
    32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
    50: "building", 51: "fence", 52: "other-structure", 60: "lane-marking", 70: "vegetation",
    71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign", 99: "other-object",
    252: "moving-car", 253: "moving-bicyclist", 254: "moving-person", 255: "moving-motorcyclist",
    256: "moving-on-rails", 257: "moving-bus", 258: "moving-truck", 259: "moving-other-vehicle"
}

def load_bin_point_cloud(file_path):
    """Load binary point cloud data from a file and return points (x, y, z)."""
    bin_pcd = np.fromfile(file_path, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, :3]  # Extract x, y, z coordinates
    return points

def save_point_cloud(points, output_path):
    """Save points as a .pcd file using Open3D."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.io.write_point_cloud(output_path, pcd)

def plot_2d_projection(points):
    """Plot 2D projection of lidar points (x, y coordinates)."""
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, c="black")  # Plot x, y coordinates
    plt.title("2D Projection of Lidar Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def load_seg_labels(file_path):
    """Load segmentation labels from a file."""
    seg_labels = np.fromfile(file_path, dtype=np.int32)
    seg_labels = seg_labels.reshape((-1, 1))
    return seg_labels

def load_detection_results(det_dir, det_file):
    """Load detection results (boxes, scores, labels) from a JSON file."""
    with open(os.path.join(det_dir, det_file)) as f:
        detections = json.load(f)
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    return boxes, scores, labels

def perform_spherical_filtering(boxes, points, seg_labels, label_dict, radius=1.0, kmeans_clusters=None):
    """Perform filtering around each detected box using spherical or K-means clustering."""
    # Extract box centers and point coordinates
    box_centers = boxes[:, :3]  # Extract box centers (x, y, z)
    point_coords = points[:, :3]  # Extract point coordinates (x, y, z)
    
    # Compute distances squared between each point and box center
    dist_squared = np.sum((point_coords[:, np.newaxis, :] - box_centers) ** 2, axis=2)
    
    # Determine the nearest box for each point (minimum distance)
    nearest_box_indices = np.argmin(dist_squared, axis=1)
    
    # Initialize filtered classes list
    filtered_classes = [[] for _ in range(len(boxes))]
    
    # Iterate over each point and assign to the nearest box based on filtering method
    for idx, (point, label) in enumerate(zip(points, seg_labels)):
        nearest_box_idx = nearest_box_indices[idx]
        box_center = box_centers[nearest_box_idx]
        
        # Check if using spherical filtering or K-means based on input arguments
        if kmeans_clusters is None:
            # Perform spherical filtering
            if np.sum((point[:3] - box_center) ** 2) <= radius**2:
                filtered_classes[nearest_box_idx].append(label_dict.get(int(label[0]), "unknown"))
        else:
			# Perform k-means clustering
            kmeans = KMeans(n_clusters=len(boxes), init=np.array(boxes), n_init=1)
            kmeans.fit(points)
			# Get cluster centers and labels
            predicted_labels = kmeans.labels_
			
            cluster_x_seg = {}
            for idp, point in enumerate(points):
                    if predicted_labels[idp] not in cluster_x_seg:
                        cluster_x_seg[predicted_labels[idp]] = []
                        cluster_x_seg[predicted_labels[idp]].append(kitti_label_dict[seg_labels[idp][0]])
                    else:
                        cluster_x_seg[predicted_labels[idp]].append(kitti_label_dict[seg_labels[idp][0]])
			
            point_cluster = kmeans_clusters[idx]
            filtered_classes[point_cluster].append(label_dict.get(int(label[0]), "unknown"))
    
    return filtered_classes

def build_polling_script(filtered_classes, labels, scores, boxes, det_x_seg_threshold=0.7, det_thresh=0.4):
    """Perform polling based on filtered classes and detection results."""
    finetuned_detections = {'boxes': [], 'labels': [], 'scores': []}
    for seg_classes, det_label, det_score, det_box in zip(filtered_classes, labels, scores, boxes):
        if len(seg_classes) > 1:
            seg_total_instances = sum(Counter(seg_classes).values())
            if det_label in seg_classes:
                seg_thresh_instance = Counter(seg_classes)[det_label] / seg_total_instances
                overall_score = (1 - det_x_seg_threshold) * seg_thresh_instance + det_score * seg_thresh_instance
                if overall_score > det_thresh:
                    finetuned_detections['boxes'].append(det_box)
                    finetuned_detections['labels'].append(det_label)
                    finetuned_detections['scores'].append(det_score)
        else:
            if det_label in seg_classes:
                finetuned_detections['boxes'].append(det_box)
                finetuned_detections['labels'].append(det_label)
                finetuned_detections['scores'].append(det_score)
    return finetuned_detections

def visualize_segmentations(points, seg_labels):
    """Visualize the point cloud with segmentation labels."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=seg_labels[:, 0], cmap='viridis')
    ax.set_title('Point Cloud with Segmentation Labels')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    # Define file paths
    bin_file_path = r"./000000.bin" # lidar .bin file location
    seg_label_file_path = r"./000000.label" # 3d segmentation file location
    det_dir = r"./pred_json/" # detection directory holding all json files with bbox, labels, and scores keys
    det_file = '000000.json' # json file name
    pcd_output_path = r"./000000.pcd" # store in pcd for easy visualization

    # Load binary point cloud data
    points = load_bin_point_cloud(bin_file_path)
    save_point_cloud(points, pcd_output_path)
    plot_2d_projection(points)

    # Load segmentation labels
    seg_labels = load_seg_labels(seg_label_file_path)

    # Perform spherical filtering
    boxes, scores, labels = load_detection_results(det_dir, det_file)
    filtered_classes = perform_spherical_filtering(boxes, points, seg_labels, kitti_label_dict)

    # Build polling script to refine detections
    finetuned_detections = build_polling_script(filtered_classes, labels, scores, boxes)

    # Print refined detections
    print("Refined Detections:")
    for idx, box in enumerate(finetuned_detections['boxes']):
        print(f"Box {idx}: Label - {finetuned_detections['labels'][idx]}, Score - {finetuned_detections['scores'][idx]}")

    # Visualize segmentations
    visualize_segmentations(points, seg_labels)

# Use the PointPillars 3D object detector from the existing OpenPCDet repo for the prediction json files.
# Use Cylinder3D repository for the segmentation results.
# this script provides the code for the KITTI dataset, please apply data adapters for NuScenes extension.

if __name__ == "__main__":
    main()
