from pydantic import BaseModel
from typing import Optional

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    status: str
    filename: str
    tracker: str
    result_url: Optional[str] = None
    error: Optional[str] = None