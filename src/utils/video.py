import cv2
from pathlib import Path
from tqdm import tqdm

def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Convert a directory of frame images into a single video file.
    
    Args:
        frames_dir (str): Path to directory containing .jpg frame files
        output_path (str): Output path for the compiled video file
        fps (int): Frames per second for the output video
        
    Returns:
        tuple: (success_flag, video_properties)
            - success_flag (bool): True if video creation succeeded
            - video_properties (tuple or None): (width, height, frame_count) if successful, None if failed
    """
    frame_files = sorted(Path(frames_dir).glob('*.jpg'))
    if not frame_files:
        return False, None
    
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_file in tqdm(frame_files, desc="Compiling Video"):
        frame = cv2.imread(str(frame_file))
        video_writer.write(frame)
    
    video_writer.release()
    return True, (width, height, len(frame_files))

def get_video_info(video_path):
    """
    Get basic properties of a video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }
