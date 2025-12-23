import shutil
import os
import uuid
from typing import Dict
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

# IMPORT THE CONFIG
from src import config
from src.trackers.yolo.inference import run_yolo
from src.trackers.tracknet.inference import run_tracknet
from src.app.schemas import TaskResponse, TaskStatus

app = FastAPI(title="Tennis Ball Tracker API")

# In-memory job store
tasks: Dict[str, dict] = {}

def process_video_task(task_id: str, video_path: Path, output_path: Path, tracker_type: str):
    """
    This function runs in the background.
    """
    try:
        print(f"Starting {tracker_type} on {video_path} (Task ID: {task_id})")
        
        if tracker_type == "tracknet":
            #saves the video to output_path
            run_tracknet(str(video_path), config.TRACKNET_MODEL_PATH, str(output_path))
        elif tracker_type == "yolo":
            #saves the video to output_path
            run_yolo(str(video_path), config.YOLO_MODEL_PATH, str(output_path))
            
        print(f"Finished processing {video_path}")
        tasks[task_id]["status"] = "completed"
        
        # Encode the filename for the URL
        import urllib.parse
        encoded_filename = urllib.parse.quote(output_path.name)
        tasks[task_id]["result_url"] = f"/download/{encoded_filename}"
        
    except Exception as e:
        print(f"Error processing video: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)

@app.post("/predict", response_model=TaskResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tracker: str = Form(...) # "yolo" or "tracknet"
):
    # 1. Validate Input
    if tracker not in ["yolo", "tracknet"]:
        raise HTTPException(status_code=400, detail="Invalid tracker type. Choose 'yolo' or 'tracknet'")

    # 2. Generate Task ID
    task_id = str(uuid.uuid4())
    
    # 3. Save the uploaded file temporarily
    input_path = config.UPLOAD_DIR / f"{task_id}_{file.filename}"
    output_filename = f"processed_{task_id}_{file.filename}"
    output_path = config.OUTPUT_DIR / output_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. Initialize Task Status
    tasks[task_id] = {
        "status": "processing",
        "filename": file.filename,
        "tracker": tracker
    }

    # 5. Add processing to background tasks
    background_tasks.add_task(process_video_task, task_id, input_path, output_path, tracker)

    # 6. Return Task ID
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Video uploaded successfully. Processing started in background."
    }

@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = config.OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)