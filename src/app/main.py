import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from .schemas import TrackingResponse

# Import our engines from Phase 1
from src.trackers.tracknet.inference import run_tracknet
from src.trackers.yolo.inference import run_yolo

app = FastAPI(title="Tennis Ball Tracker API")

# Setup directories for storage
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define paths to your weights (Update these paths to match your actual file locations!)
TRACKNET_WEIGHTS = "src/trackers/tracknet/weights/model_best.pth.tar" 
YOLO_WEIGHTS = "src/trackers/yolo/weights/best.pt"

def process_video_task(video_path: Path, output_path: Path, tracker_type: str):
    """
    This function runs in the background.
    """
    try:
        print(f"Starting {tracker_type} on {video_path}")
        
        if tracker_type == "tracknet":
            # We assume run_tracknet saves the video to output_path
            run_tracknet(str(video_path), TRACKNET_WEIGHTS, str(output_path))
        elif tracker_type == "yolo":
            # We assume run_yolo saves the video to output_path
            run_yolo(str(video_path), YOLO_WEIGHTS, str(output_path))
            
        print(f"Finished processing {video_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        # In a real app, you'd update a database status here

@app.post("/predict", response_model=TrackingResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tracker: str = Form(...) # "yolo" or "tracknet"
):
    # 1. Validate Input
    if tracker not in ["yolo", "tracknet"]:
        raise HTTPException(status_code=400, detail="Invalid tracker type. Choose 'yolo' or 'tracknet'")

    # 2. Save the uploaded file temporarily
    input_path = UPLOAD_DIR / file.filename
    output_filename = f"processed_{file.filename}"
    output_path = OUTPUT_DIR / output_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Add processing to background tasks
    background_tasks.add_task(process_video_task, input_path, output_path, tracker)

    # 4. Return immediate response
    return {
        "filename": file.filename,
        "tracker_type": tracker,
        "status": "processing",
        "message": "Video uploaded successfully. Processing started in background.",
        "download_url": f"/download/{output_filename}"
    }

@app.get("/download/{filename}")
async def download_video(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return {"error": "File not found or processing not complete."}