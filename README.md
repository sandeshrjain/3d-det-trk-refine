# 3d-det-trk-refine
3D Object Detection and Tracking Refinement with Ensemble Methods and Spatiotemporal Filtering

# 3D Object Detection and Tracking Refinement

This repository contains code for refining 3D object detection results and improving 3D object tracking using state-of-the-art methods.

## Detection Refinement

### Description

The `official_det_refine.py` script refines 3D object detection results obtained from OpenPCDet and cylinder3d detectors. It implements spherical and K-means filtering to enhance the accuracy of detected objects.

### Dependencies

To run the detection refinement script, you'll need to install the following dependencies:

- OpenPCDet: https://github.com/open-mmlab/OpenPCDet
  - Follow the installation instructions in the OpenPCDet repository to set up the environment and download pre-trained models.

- Cylinder3D: https://github.com/xinge008/Cylinder3D
  - Install Cylinder3D and download the appropriate pre-trained models for 3D object detection.

### Usage

1. Install OpenPCDet and Cylinder3D as per their respective repositories.
2. Download pre-trained models and configure paths in the `detection_refinement.py` script.
3. Run `official_det_refine.py` to refine detection results using spherical and K-means filtering.

## Tracker Refinement and Evaluation

### Description

The `official_adrit.py` script improves 3D object tracking using AB3DMOT (Association-Based 3D Multi-Object Tracking). It refines object trajectories and evaluates tracking performance.

### Dependencies

To run the tracker refinement and evaluation script, you'll need to install:

- AB3DMOT: https://github.com/xinshuoweng/AB3DMOT
  - Set up AB3DMOT environment and download required datasets and models.

### Usage

1. Install AB3DMOT by following the instructions in the AB3DMOT repository.
2. Configure paths and parameters in the `tracker_refinement.py` script.
3. Run `official_adrit.py` to refine object trajectories and evaluate tracking performance.
