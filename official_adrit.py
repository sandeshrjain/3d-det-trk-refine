# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:02:04 2023
@author: Anonymos
"""

import os
import numpy as np
from copy import deepcopy
import json


# Global configuration
ids_repair = False
look_ahead = 5 # number of frames to consider when mathcing tracking IDs
file_path = r'tracking_file_path.txt' # your 3D MOT tracking file before refinement (using AB3DMOT KITTI tracking format)
path_save = r'path_to_save_output_reid'
write_json = True #write the refined tracking results to json file if true else to .txt
use_3d_dist = True # use 3d distance for track similar ity or 3D IoU
thresh_iou = 0.2 # IoU threshold if used


def load_all_tracks(file_path):
    """
    Load all tracks from the specified file into a dictionary of frames.

    Args:
        file_path (str): Path to the file containing track information.

    Returns:
        list[dict]: List of dictionaries where each dictionary represents a frame
                    with track IDs as keys and track information as values.
    """
    all_tracks = []

    with open(file_path, 'r') as f:
        total_frames = int(f.readline().split(' ')[0]) + 1  # Zero-indexed, so +1
        f.seek(0)  # Reset file pointer to start

        for line in f:
            line = line.split(' ')
            if line[2] == 'Car':  # For KITTI, change to 'Car'
                frame = int(line[0])
                track_id = int(line[1])
                if frame >= len(all_tracks):
                    all_tracks.extend([{} for _ in range(frame - len(all_tracks) + 1)])
                all_tracks[frame][track_id] = line
        print("Total lidar frames: ", total_frames)
    return all_tracks


def find_drops_and_adds(all_tracks):
    """
    Find dropped and added track IDs between consecutive frames.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.

    Returns:
        list[set], list[set]: Drops and Adds lists where each element represents the dropped or added
                              track IDs for the corresponding frame.
    """
    total_frames = len(all_tracks)
    Drops = [set() for _ in range(total_frames)]
    Adds = [set() for _ in range(total_frames)]

    for idx in range(total_frames - 1):
        current_frame_tracks = set(all_tracks[idx].keys())
        next_frame_tracks = set(all_tracks[idx + 1].keys())
        Drops[idx + 1].update(current_frame_tracks - (current_frame_tracks & next_frame_tracks))
        Adds[idx + 1].update(next_frame_tracks - (current_frame_tracks & next_frame_tracks))

    return Drops, Adds


def interpolate_missing_tracks(all_tracks, Drops, Adds):
    """
    Interpolate missing tracks based on Drops and Adds information.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.
        Drops (list[set]): Drops list where each element represents the dropped track IDs for the frame.
        Adds (list[set]): Adds list where each element represents the added track IDs for the frame.

    Returns:
        list[dict]: Updated all_tracks after performing interpolation.
    """
    for idx, dropped_tracks in enumerate(Drops):
        if len(dropped_tracks) > 0:
            for track_id in dropped_tracks:
                try:
                    line_track_id = all_tracks[idx - 1][track_id]
                    for add_idx, added_tracks in enumerate(Adds[idx: idx + look_ahead]):
                        if len(added_tracks) > 0:
                            for add_track_id in added_tracks:
                                add_track_info = all_tracks[idx + add_idx][add_track_id]
                                if within_bbox(line_track_id, add_track_info):
                                    interpolate_track(all_tracks, line_track_id, add_track_info, idx, add_idx)
                                    if ids_repair:
                                        repair_ids(all_tracks, add_track_id, line_track_id, idx, add_idx)
                                    break
                except KeyError:
                    print('KeyError occurred in outer loop')
                    pass

    return all_tracks


def within_bbox(sline, cline, threshold=1):
    """
    Check if the bounding box centers are close enough.

    Args:
        sline (list): Information about the source track.
        cline (list): Information about the candidate track.
        threshold (float): Maximum distance threshold for the center points.

    Returns:
        bool: True if the centers are within the threshold distance, False otherwise.
    """
	
    if use_3d_dist:
	
        s_center = np.array([float(sline[i]) for i in range(13, 16)])
        c_center = np.array([float(cline[i]) for i in range(13, 16)])
        return np.all(np.abs(s_center - c_center) < threshold)
	
    else:
        box1 = np.array([float(sline[i]) for i in range(10, 16)])
        box2 = np.array([float(sline[i]) for i in range(10, 16)])
        iou_3d = compute_3diou(box1, box2)
        return iou_3d > thresh_iou
	
    

def compute_3diou(box1, box2):
    """
    Compute the 3D Intersection over Union (3DIoU) between two 3D bounding boxes.

    Args:
        box1 (list): List containing [x1, y1, z1, h1, w1, l1, a1], where:
                     (x1, y1, z1) is the center of the box,
                     h1 is the height,
                     w1 is the width,
                     l1 is the length,
                     a1 is the orientation angle (radians).
        box2 (list): List containing [x2, y2, z2, h2, w2, l2, a2], where:
                     (x2, y2, z2) is the center of the box,
                     h2 is the height,
                     w2 is the width,
                     l2 is the length,
                     a2 is the orientation angle (radians).

    Returns:
        float: 3DIoU value between the two boxes.
    """
    x1, y1, z1, h1, w1, l1, a1 = box1
    x2, y2, z2, h2, w2, l2, a2 = box2

    # Calculate box dimensions
    vol1 = h1 * w1 * l1
    vol2 = h2 * w2 * l2

    # Calculate box corners for box1
    corners1 = get_box_corners(x1, y1, z1, h1, w1, l1, a1)
    corners2 = get_box_corners(x2, y2, z2, h2, w2, l2, a2)

    # Calculate intersection box
    int_box = get_intersection_box(corners1, corners2)

    if int_box is None:
        return 0.0

    # Calculate intersection volume
    int_vol = (int_box[1][0] - int_box[0][0]) * (int_box[1][1] - int_box[0][1]) * (int_box[1][2] - int_box[0][2])

    # Calculate union volume
    union_vol = vol1 + vol2 - int_vol

    # Calculate IoU
    iou = int_vol / union_vol if union_vol > 0 else 0.0

    return iou


def get_box_corners(x, y, z, h, w, l, a):
    """
    Compute the corners of a 3D bounding box.

    Args:
        x (float): x-coordinate of the box center.
        y (float): y-coordinate of the box center.
        z (float): z-coordinate of the box center.
        h (float): Height of the box.
        w (float): Width of the box.
        l (float): Length of the box.
        a (float): Orientation angle of the box (radians).

    Returns:
        list: List of 8 corners (x, y, z) of the box.
    """
    # Get half dimensions
    h, hw, hl = h / 2, w / 2, l / 2

    # Define rotation matrix
    rot_matrix = np.array([[np.cos(a), -np.sin(a), 0],
                            [np.sin(a), np.cos(a), 0],
                            [0, 0, 1]])

    # Define box corners in the object coordinate system
    corners = np.array([[-hw, -hl, 0], [hw, -hl, 0], [hw, hl, 0], [-hw, hl, 0],
                        [-hw, -hl, h], [hw, -hl, h], [hw, hl, h], [-hw, hl, h]])

    # Rotate box corners and translate to global coordinate system
    corners = np.dot(corners, rot_matrix.T) + np.array([x, y, z])

    return corners


def get_intersection_box(corners1, corners2):
    """
    Calculate the intersection box given the corners of two boxes.

    Args:
        corners1 (np.array): Corners of the first box (8x3 array).
        corners2 (np.array): Corners of the second box (8x3 array).

    Returns:
        list: List containing [[min_x, min_y, min_z], [max_x, max_y, max_z]] of the intersection box.
              Returns None if there is no intersection.
    """
    int_min = np.maximum(np.min(corners1, axis=0), np.min(corners2, axis=0))
    int_max = np.minimum(np.max(corners1, axis=0), np.max(corners2, axis=0))

    if np.any(int_min >= int_max):
        return None

    return [int_min.tolist(), int_max.tolist()]


def interpolate_track(all_tracks, line_track_id, add_track_info, start_idx, add_idx):
    """
    Interpolate track information between two frames.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.
        line_track_id (list): Information about the track from the previous frame.
        add_track_info (list): Information about the added track in the current frame.
        start_idx (int): Starting frame index.
        add_idx (int): Index of the frame where the track is added.
    """
    interpolation_length = int(add_track_info[0]) - int(line_track_id[0]) - 1
    for idk, framek in enumerate(all_tracks[int(line_track_id[0]) + 1: int(add_track_info[0])]):
        frame_index = int(line_track_id[0]) + 1 + idk
        last_missing_key = int(add_track_info[1])
        all_tracks[frame_index][last_missing_key] = deepcopy(line_track_id)
        interpolate_values(all_tracks[frame_index][last_missing_key], line_track_id, add_track_info, idk, interpolation_length)


def interpolate_values(interpolated_track_info, line_track_id, add_track_info, idk, interpolation_length):
    """
    Interpolate values between two tracks.

    Args:
        interpolated_track_info (list): Interpolated track information to be updated.
        line_track_id (list): Information about the track from the previous frame.
        add_track_info (list): Information about the added track in the current frame.
        idk (int): Index of the current frame in the interpolation sequence.
        interpolation_length (int): Total number of frames to interpolate between the tracks.
    """
    keys_to_interpolate = [13, 14, 15, 16]  # x, y, z, yaw
    for key_index, key in enumerate(keys_to_interpolate):
        start_value = float(line_track_id[key])
        end_value = float(add_track_info[key])
        interpolated_value = start_value + (idk + 1) * (end_value - start_value) / (interpolation_length + 1)
        interpolated_track_info[key] = interpolated_value


def repair_ids(all_tracks, add_track_id, line_track_id, start_idx, add_idx):
    """
    Repair track IDs to prevent ID switching during interpolation.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.
        add_track_id (int): ID of the track to be repaired.
        line_track_id (list): Information about the track from the previous frame.
        start_idx (int): Starting frame index.
        add_idx (int): Index of the frame where the track is added.
    """
    for idr, framer in enumerate(all_tracks[min(int(add_track_id[0]) + 1, len(all_tracks) - 1):]):
        if int(add_track_id[0]) in framer:
            first_change = min(int(add_track_id[0]) + 1, len(all_tracks) - 1) + idr
            try:
                all_tracks[first_change][int(line_track_id[1])] = deepcopy(all_tracks[first_change][int(add_track_id[1])])
                del all_tracks[first_change][int(add_track_id[1])]
            except KeyError:
                break
        else:
            break


def save_interpolated_tracks_to_json(all_tracks, path, names_files):
    """
    Save interpolated tracks to JSON files.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.
        path (str): Path to save the JSON files.
        names_files (list): List of filenames.
    """
    for idx, frame in enumerate(all_tracks):
        jfile = {'boxes': [], 'labels': [], 'scores': [], 'trks': []}
        for key, value in frame.items():
            [h, w, l, x, y, z, a, s] = [float(x) for x in value[10:]]
            jfile['boxes'].append([x, y, z, h, w, l, a, 0, 0])
            jfile['labels'].append(0)
            jfile['scores'].append(s)
            jfile['trks'].append(value[1])

        json_object = json.dumps(jfile, indent=4)
        write_path_json = os.path.join(path, names_files[idx][:-4] + '.json')
        with open(write_path_json, "w") as outfile:
            outfile.write(json_object)


def save_interpolated_tracks_to_txt(all_tracks, txt_path):
    """
    Save interpolated tracks to text files.

    Args:
        all_tracks (list[dict]): List of dictionaries where each dictionary represents a frame
                                  with track IDs as keys and track information as values.
        txt_path (str): Path to save the text file.
    """
    cvt_det = []
    for idx, frame in enumerate(all_tracks):
        for key, value in frame.items():
            [h, w, l, x, y, z, a, s] = [float(x) for x in value[10:]]
            l_ = 'Car'
            frame_id = deepcopy(idx)
            w_line = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
                int(frame_id), int(value[1]), l_, -1.8, 1, 2, 3, 4, h, w, l, x, y, z, a, s)
            cvt_det.append(w_line)

    n_cvt = np.asarray(cvt_det)
    write_path_txt = os.path.join(txt_path, 'refined.txt')
    np.savetxt(write_path_txt, n_cvt, fmt=('%s'), delimiter=' ')




def main():
    all_tracks = load_all_tracks(file_path)
    Drops, Adds = find_drops_and_adds(all_tracks)
    all_tracks = interpolate_missing_tracks(all_tracks, Drops, Adds)
    
    # Optional: Save the interpolated tracks to JSON or text files
    if write_json:
        save_interpolated_tracks_to_json(all_tracks, path_save, 'refined.json')

    else:
        save_interpolated_tracks_to_txt(all_tracks, './')


# For evaluation use the utilities present in the official AB3DMOT GitHub repository


if __name__ == "__main__":
    main()

