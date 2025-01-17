from pydantic import BaseModel
from typing import List

class ImageMatch(BaseModel):
    image_data: str
    similarity_score: float
    file_path: str

class Match(BaseModel):
    image_data: str
    similarity_score: float
    file_path: str

class SearchResponse(BaseModel):
    matches: List[Match]
    total_matches: int
    page: int
    page_size: int
    total_pages: int

class HealthResponse(BaseModel):
    status: str
    version: str

