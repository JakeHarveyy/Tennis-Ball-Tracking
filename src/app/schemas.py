from pydantic import BaseModel
from typing import Optional

class TrackingResponse(BaseModel):
    filename: str
    tracker_type: str
    status: str
    message: str
    download_url: Optional[str] = None