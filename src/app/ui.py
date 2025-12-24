import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

import streamlit as st
import requests
import time
import subprocess
import os
from src import config

# Configuration
API_URL = config.API_URL

def convert_to_h264(input_path, output_path):
    """
    Converts a video to H.264 format (browser compatible) using FFmpeg.
    """
    # ffmpeg -i input.mp4 -vcodec libx264 -acodec aac output.mp4
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p', # Critical for browser playback
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your PATH to enable in-browser video playback.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")

st.set_page_config(page_title="Tennis Tracker", page_icon="🎾", layout="wide")
st.title("🎾 Tennis Ball Tracker")

# Example Section
st.subheader("Project Demo")
st.markdown("Here is an example of the raw input and the resulting tracking outputs.")

example_dir = root_path / "data" / "test_clip" / "example"
raw_clip = example_dir / "Clip1.mp4"
tracknet_clip = example_dir / "Clip1_tracknet_trajectory.mp4"
yolo_clip = example_dir / "Clip1_yolo_trajectory.mp4"

def get_display_video_path(video_path):
    """Checks for H.264 version, converts if missing, returns path string."""
    if not video_path.exists():
        return None
    
    # Define the target H.264 path
    h264_path = video_path.with_name(f"{video_path.stem}_h264{video_path.suffix}")
    
    if not h264_path.exists():
        # Convert if it doesn't exist
        try:
            convert_to_h264(str(video_path), str(h264_path))
        except Exception as e:
            st.error(f"Could not convert {video_path.name}: {e}")
            return None
            
    return str(h264_path)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Raw Input")
    v_path = get_display_video_path(raw_clip)
    if v_path:
        st.video(v_path, autoplay=True, loop=True, muted=True)
    else:
        st.error("File not found")

with col2:
    st.markdown("### TrackNet Output")
    v_path = get_display_video_path(tracknet_clip)
    if v_path:
        st.video(v_path, autoplay=True, loop=True, muted=True)
    else:
        st.error("File not found")

with col3:
    st.markdown("### YOLO Output")
    v_path = get_display_video_path(yolo_clip)
    if v_path:
        st.video(v_path, autoplay=True, loop=True, muted=True)
    else:
        st.error("File not found")

st.divider()

# sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    tracker_type = st.selectbox(
        "Select Tracker Model",
        ("yolo", "tracknet"),
        help="test"
    )

# File Upload
uploaded_file = st.file_uploader("Choose a tennis video", type=["mp4", "mov", "avi"])

# The Main Logic
if uploaded_file is not None:
    # Display video, to show loaded
    st.video(uploaded_file)

    # Trigger Button
    if st.button("Start Tracking"):
        # Prepare payload (input params)
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {"tracker": tracker_type}

        # Send to API
        with st.spinner("Uploading video to server..."):
            try:
                response = requests.post(f"{API_URL}/predict", files=files, data=data)
                response.raise_for_status()
                task_data = response.json()
                task_id = task_data["task_id"]
                st.success(f"Processing started! Task ID: {task_id}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
                st.stop()
        
        # Polling Loop
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        while True:
            # Ask API for status
            status_res = requests.get(f"{API_URL}/tasks/{task_id}")
            status_data = status_res.json()
            status = status_data["status"]

            if status == "completed":
                progress_bar.progress(100)
                status_placeholder.success("Tracking Inference Completed!")

                # Construct Download URL
                result_url = f"{API_URL}{status_data['result_url']}"
                
                # Download the video content
                with st.spinner("Downloading and converting video for browser playback..."):
                    try:
                        # 1. Download from API
                        video_res = requests.get(result_url)
                        video_res.raise_for_status()
                        
                        # 3. Convert to H.264
                        # We use a unique filename to avoid conflicts if multiple users are running
                        import uuid
                        unique_id = str(uuid.uuid4())
                        temp_input = f"temp_in_{unique_id}.mp4"
                        temp_output = f"temp_out_{unique_id}.mp4"

                        with open(temp_input, "wb") as f:
                            f.write(video_res.content)

                        convert_to_h264(temp_input, temp_output)
                        
                        # 4. Display Result
                        st.subheader("Tracking Result")
                        if os.path.exists(temp_output):
                            with open(temp_output, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                        else:
                            st.warning("Video conversion failed. You can still download the result below.")
                            
                        # Cleanup
                        if os.path.exists(temp_input):
                            os.remove(temp_input)
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                            
                    except Exception as e:
                        st.error(f"Error processing video for display: {e}")
                        # Fallback to link
                        st.markdown(f"### [Download Result Video]({result_url})")
                break
            
            elif status == "failed":
                status_placeholder.error(f"Task Failed: {status_data.get('error')}")

            else:
                # Still Processing
                status_placeholder.info(f"Status: {status}... (Please Wait)")
                time.sleep(2)
