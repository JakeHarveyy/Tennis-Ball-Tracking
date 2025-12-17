import torch
import cv2
import numpy as np
import collections
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from .model import TrackNet

INPUT_WIDTH = 640
INPUT_HEIGHT = 360
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def postprocess(heatmap):
    """
    Returns center_x, center_y, and radius from heatmap.
    """
    heatmap = heatmap.astype(np.uint8)
    ret, binary_heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(binary_heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                               param1=50, param2=2, minRadius=2, maxRadius=7)
    
    if circles is not None and len(circles) == 1:
        x = circles[0][0][0]
        y = circles[0][0][1]
        r = circles[0][0][2]
        return x, y, r
    
    return None, None, None

def run_tracknet(video_path: str, model_path: str, output_video_path: str = None):
    """
    Run TrackNet inference on a video.
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to TrackNet model weights
        output_video_path (str, optional): Path to save annotated video
        
    Returns:
        pd.DataFrame: DataFrame containing tracking predictions
    """
    # Load model
    model = TrackNet().to(DEVICE)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file at {video_path}")

    # Extract video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if output path is provided
    video_writer = None
    if output_video_path:
        # Use vp09 (VP9) which is browser-friendly and doesn't require external DLLs on Windows
        fourcc = cv2.VideoWriter_fourcc(*'vp09')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (original_width, original_height))

    # Initialize frame buffer and prediction storage
    frame_buffer = collections.deque(maxlen=3)
    predictions_list = []
    frame_id_counter = 0

    print(f"Processing video: {total_frames} frames, {original_width}x{original_height}")
    pbar = tqdm(total=total_frames, desc="TrackNet Inference")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to model input dimensions
        resized_frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        frame_buffer.append(resized_frame)
        
        current_pred = {
            'frame_id': frame_id_counter,
            'tracknet_x': np.nan,
            'tracknet_y': np.nan,
            'tracknet_w': np.nan,
            'tracknet_h': np.nan
        }

        # Process when we have 3 consecutive frames
        if len(frame_buffer) == 3:
            # Convert frames to tensor format
            imgs_list = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).float() / 255.0 for f in frame_buffer]
            input_tensor = torch.cat([t.permute(2, 0, 1) for t in imgs_list], dim=0).unsqueeze(0).to(DEVICE)
            
            # Run model inference
            with torch.no_grad():
                predictions = model(input_tensor)
            
            # Extract ball coordinates
            pred_heatmap = torch.argmax(predictions.squeeze(0), dim=0).cpu().numpy()
            px, py, pr = postprocess(pred_heatmap)
            
            if px is not None:
                scale_x = original_width / INPUT_WIDTH
                scale_y = original_height / INPUT_HEIGHT
                
                center_x = px * scale_x
                center_y = py * scale_y
                radius = pr * ((scale_x + scale_y) / 2.0)
                
                current_pred.update({
                    'tracknet_x': int(center_x - radius),
                    'tracknet_y': int(center_y - radius),
                    'tracknet_w': 20,
                    'tracknet_h': 20
                })
                
                if video_writer:
                     cv2.circle(frame, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)

        predictions_list.append(current_pred)
        
        if video_writer:
            video_writer.write(frame)
            
        frame_id_counter += 1
        pbar.update(1)
    
    cap.release()
    if video_writer:
        video_writer.release()
        
    return pd.DataFrame(predictions_list)
