import logging
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services.image_service import ImageSearchService
from app.models.schemas import ImageMatch, HealthResponse, SearchResponse
from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FastAPI App")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; consider restricting for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize service
service = ImageSearchService(settings.IMAGES_DIR)


# Handle both `/search` and `/search/`
@app.post("/search", response_model=SearchResponse)
@app.post("/search/", response_model=SearchResponse)
async def search_faces(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=settings.MAX_PAGE_SIZE, description="Results per page")
):
    logger.info(f"Received search request: file={file.filename}, page={page}, page_size={page_size}")
    try:
        # Pass the file and query parameters to the service
        response = await service.find_similar_faces(file, page, page_size)
        logger.info(f"Search completed successfully: {len(response.matches)} matches found.")
        return response
    except Exception as e:
        logger.error(f"Error during search_faces: {e}", exc_info=True)
        # Return error response with detailed exception message
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )



@app.get("/health", response_model=HealthResponse)
def health_check():
    """Basic health check endpoint"""
    logger.info("Health check endpoint called.")
    return {
        "status": "healthy",
        "version": settings.API_VERSION
    }


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app with hot reload enabled
    logger.info("Starting FastAPI application.")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
