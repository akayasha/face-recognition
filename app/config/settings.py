from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Base directory for images
    IMAGES_DIR: str = "C:\images"

    # API Settings
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "Face Recognition API"

    # Face Recognition Settings
    SIMILARITY_THRESHOLD: float = 0.6
    MAX_PAGE_SIZE: int = 50

    # Image Settings
    SUPPORTED_FORMATS: set = {'.jpg', '.jpeg', '.png'}

    class Config:
        env_file = ".env"


settings = Settings()