from pathlib import Path
import face_recognition
import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple
from fastapi import UploadFile
from app.config.settings import settings
from app.models.schemas import SearchResponse, Match


class ImageSearchService:
    def __init__(self, base_image_path: str):
        self.base_path = Path(base_image_path)
        self.supported_formats = settings.SUPPORTED_FORMATS

    def get_all_images(self) -> List[Path]:
        """Recursively get all image files from the base directory."""
        image_files = []
        for path in self.base_path.rglob('*'):
            if path.suffix.lower() in self.supported_formats:
                image_files.append(path)
        return image_files

    def get_face_encoding(self, image_path: Path) -> List[np.ndarray]:
        """Get face encodings from an image file."""
        try:
            image = face_recognition.load_image_file(str(image_path))
            encodings = face_recognition.face_encodings(image)
            return encodings
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return []

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def find_similar_faces(
            self,
            query_image: UploadFile,
            page: int = 1,
            page_size: int = 10
    ) -> Dict:
        """Find similar faces in the image directory with pagination."""
        contents = await query_image.read()
        query_img = Image.open(io.BytesIO(contents))
        query_img_array = np.array(query_img)
        query_encodings = face_recognition.face_encodings(query_img_array)

        if not query_encodings:
            return self._create_empty_response(page, page_size)

        query_encoding = query_encodings[0]
        results = self._search_images(query_encoding)
        return self._create_paginated_response(results, page, page_size)

    def _search_images(self, query_encoding: np.ndarray) -> List[Tuple[float, Path, float]]:
        """Search through images and return matches with scores."""
        results = []
        for image_path in self.get_all_images():
            encodings = self.get_face_encoding(image_path)
            for encoding in encodings:
                distance = face_recognition.face_distance([query_encoding], encoding)[0]
                if distance < settings.SIMILARITY_THRESHOLD:
                    similarity_score = 1 - distance
                    results.append((distance, image_path, similarity_score))
        results.sort(key=lambda x: x[0])
        return results

    def _create_paginated_response(
            self,
            results: List[Tuple[float, Path, float]],
            page: int,
            page_size: int
    ) -> Dict:
        """Create paginated response with base64 images."""
        total_matches = len(results)
        total_pages = (total_matches + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_matches = []
        for _, image_path, similarity_score in results[start_idx:end_idx]:
            try:
                base64_image = self.image_to_base64(image_path)
                paginated_matches.append({
                    "image_data": base64_image,
                    "similarity_score": round(similarity_score * 100, 2),
                    "file_path": str(image_path)
                })
            except Exception as e:
                print(f"Error processing result {image_path}: {str(e)}")

        return {
            "matches": paginated_matches,
            "total_matches": total_matches,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }

    def _create_empty_response(self, page: int, page_size: int) -> Dict:
        """Create empty response when no faces are found."""
        return {
            "matches": [],
            "total_matches": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0
        }

    def _create_paginated_response(
            self,
            results: List[Tuple[float, Path, float]],
            page: int,
            page_size: int
    ) -> SearchResponse:
        """Create paginated response with base64 images."""
        total_matches = len(results)
        total_pages = (total_matches + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_matches = []
        for _, image_path, similarity_score in results[start_idx:end_idx]:
            try:
                base64_image = self.image_to_base64(image_path)
                paginated_matches.append(Match(
                    image_data=base64_image,
                    similarity_score=round(similarity_score * 100, 2),
                    file_path=str(image_path)
                ))
            except Exception as e:
                print(f"Error processing result {image_path}: {str(e)}")

        return SearchResponse(
            matches=paginated_matches,
            total_matches=total_matches,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    def _create_empty_response(self, page: int, page_size: int) -> SearchResponse:
        """Create empty response when no faces are found."""
        return SearchResponse(
            matches=[],
            total_matches=0,
            page=page,
            page_size=page_size,
            total_pages=0
        )
