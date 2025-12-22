import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

import streamlit as st
import requests
import time
from src import config

# Configuration
API_URL = config.API_URL

st.set_page_config(page_title="Tennis Tracker", page_icon="🎾")
st.title("🎾 Tennis Ball Tracker")

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

                # Display Result
                st.subheader("Tracking Result")
                st.video(result_url)
                break
            
            elif status == "failed":
                status_placeholder.error(f"Task Failed: {status_data.get('error')}")

            else:
                # Still Processing
                status_placeholder.info(f"Status: {status}... (Please Wait)")
                time.sleep(2)
