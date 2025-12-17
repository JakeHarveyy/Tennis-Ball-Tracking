from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def run_yolo(video_path: str, model_path: str, output_path: str, conf_threshold: float = 0.3):
    """
    Runs YOLO tracking on a video and saves the output with trajectory.
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to YOLO model weights
        output_path (str): Path to save annotated video
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        list: List of trajectory points (x, y)
    """
    model = YOLO(model_path)
    
    # Open video to get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file at {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Initialize video writer
    # Use vp09 (VP9) which is browser-friendly and doesn't require external DLLs on Windows
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Run tracking
    results = model.track(source=video_path, stream=True, persist=True)
    
    trajectory_points = []
    
    print(f"Processing video with YOLO: {video_path}")
    
    for r in tqdm(results, total=total_frames, desc="YOLO Inference"):
        annotated_frame = r.plot()
        
        if len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            
            best_box_idx = confidences.argmax()
            
            if confidences[best_box_idx] >= conf_threshold:
                best_box = boxes[best_box_idx]
                center_x = (best_box[0] + best_box[2]) // 2
                center_y = (best_box[1] + best_box[3]) // 2
                trajectory_points.append((center_x, center_y))
        
        # Draw trajectory
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                cv2.line(annotated_frame, 
                        trajectory_points[i - 1], 
                        trajectory_points[i], 
                        (0, 0, 255), 3)
        
        video_writer.write(annotated_frame)
        
    video_writer.release()
    return trajectory_points
