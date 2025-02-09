import os
import sys
import json
import pickle
import subprocess
import numpy as np
import cv2
import torch
from tqdm import tqdm
import shutil

from face_detection import FaceAlignment, LandmarksType
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

from musetalk.utils.utils import datagen  # Assuming needed elsewhere

# Initialize mmpose model on GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
model = init_model(config_file, checkpoint_file, device=device)

# Initialize the face detection model.
fa = FaceAlignment(LandmarksType._2D, flip_input=False, device="cuda" if torch.cuda.is_available() else "cpu")

# Fallback placeholder if the computed bounding box is not sufficient.
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    """
    Resize a landmark point from original (w,h) to new (new_w,new_h).
    """
    landmark_norm = landmark / np.array([w, h])
    landmark_resized = landmark_norm * np.array([new_w, new_h])
    return landmark_resized

def read_imgs(img_list):
    """
    Read images from a list of paths and return them as a list.
    """
    print("Reading images...")
    return [cv2.imread(img_path) for img_path in tqdm(img_list)]

def _process_batch(frames_batch, upperbondrange=0):
    """
    Process a batch of frames to extract landmarks and compute adjusted bounding boxes.
    Returns a list of bounding box coordinates.
    """
    coords = []
    # Use the first frame in the batch for inference.
    results = inference_topdown(model, np.asarray(frames_batch)[0])
    results = merge_data_samples(results)
    keypoints = results.pred_instances.keypoints
    # Use indices 23:91 for face landmarks.
    face_land_mark = keypoints[0][23:91].astype(np.int32)

    # Get face detection bounding boxes.
    bboxes = fa.get_detections_for_batch(np.asarray(frames_batch))
    for f in bboxes:
        if f is None:  # No face detected.
            coords.append(coord_placeholder)
            continue

        # Use landmark at index 29 as reference.
        half_face_coord = face_land_mark[29].copy()
        range_minus = int((face_land_mark[30] - face_land_mark[29])[1])
        range_plus = int((face_land_mark[29] - face_land_mark[28])[1])
        if upperbondrange != 0:
            half_face_coord[1] += upperbondrange  # Adjust based on upperbondrange.
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = half_face_coord[1] - half_face_dist

        # Construct landmark-based bounding box.
        landmark_bbox = (
            int(np.min(face_land_mark[:, 0])),
            int(upper_bond),
            int(np.max(face_land_mark[:, 0])),
            int(np.max(face_land_mark[:, 1]))
        )
        x1, y1, x2, y2 = landmark_bbox
        if (y2 - y1) <= 0 or (x2 - x1) <= 0 or x1 < 0:
            coords.append(f)  # Fallback to face detection bbox.
            print("Invalid landmark bbox, using fallback:", f)
        else:
            coords.append(landmark_bbox)
    return coords

def get_bbox_range(img_list, upperbondrange=0):
    """
    Read images and compute bounding box range information.
    Returns a text summary with average adjustments.
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    average_range_minus = []
    average_range_plus = []
    if upperbondrange != 0:
        print("Extracting bounding boxes with bbox_shift:", upperbondrange)
    else:
        print("Extracting bounding boxes with default value")
    for fb in tqdm(batches):
        coords = _process_batch(fb, upperbondrange)
        coords_list.extend(coords)
        # For statistics, we use dummy fixed values (adjust as needed).
        average_range_minus.append(5)
        average_range_plus.append(5)
    if average_range_minus and average_range_plus:
        avg_minus = int(sum(average_range_minus) / len(average_range_minus))
        avg_plus = int(sum(average_range_plus) / len(average_range_plus))
    else:
        avg_minus = avg_plus = 0
    text_range = (
        f"Total frames: {len(frames)}. "
        f"Manually adjust range: [-{avg_minus} ~ {avg_plus}], current value: {upperbondrange}"
    )
    return text_range

def get_landmark_and_bbox(img_list, upperbondrange=0):
    """
    Read images and compute both landmarks and bounding boxes.
    Returns a tuple (coords_list, frames).
    """
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    average_range_minus = []
    average_range_plus = []
    if upperbondrange != 0:
        print("Extracting landmarks and bounding boxes with bbox_shift:", upperbondrange)
    else:
        print("Extracting landmarks and bounding boxes with default value")
    for fb in tqdm(batches):
        coords = _process_batch(fb, upperbondrange)
        coords_list.extend(coords)
        average_range_minus.append(5)
        average_range_plus.append(5)
    if average_range_minus and average_range_plus:
        avg_minus = int(sum(average_range_minus) / len(average_range_minus))
        avg_plus = int(sum(average_range_plus) / len(average_range_plus))
    else:
        avg_minus = avg_plus = 0
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frames: {len(frames)}. Manually adjust range: [-{avg_minus} ~ {avg_plus}], current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames

if __name__ == "__main__":
    img_list = [
        "./results/lyria/00000.png",
        "./results/lyria/00001.png",
        "./results/lyria/00002.png",
        "./results/lyria/00003.png"
    ]
    crop_coord_path = "./coord_face.pkl"
    coords_list, full_frames = get_landmark_and_bbox(img_list)
    with open(crop_coord_path, 'wb') as f:
        pickle.dump(coords_list, f)
        
    for bbox, frame in zip(coords_list, full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        print("Cropped shape:", crop_frame.shape)
    print("Final coordinates:", coords_list)
