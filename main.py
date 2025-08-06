"""
FastAPI-based Plagiarism Detection System for Bangla Language
Provides text similarity scoring and plagiarism detection with webhook integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import os

from schemas.request_models import PlagiarismRequest, WebhookConfig
from schemas.response_models import PlagiarismResponse, SimilarityResult, HealthResponse
from models.plagiarism_detector import PlagiarismDetector
from services.webhook_service import WebhookService
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bangla Plagiarism Detection API",
    description="ML-powered plagiarism detection system for Bangla language text with similarity scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = Settings()
plagiarism_detector = PlagiarismDetector()
webhook_service = WebhookService()

@app.on_event("startup")
async def startup_event():
    """Initialize ML model on startup"""
    logger.info("Starting Bangla Plagiarism Detection API...")
    try:
        await plagiarism_detector.initialize()
        logger.info("ML model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML model: {str(e)}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Bangla Plagiarism Detection API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with model status"""
    model_status = "loaded" if plagiarism_detector.is_initialized else "not_loaded"
    return HealthResponse(
        status="healthy",
        message=f"API is running, ML model status: {model_status}",
        version="1.0.0"
    )

@app.post("/detect-plagiarism", response_model=PlagiarismResponse)
async def detect_plagiarism(
    request: PlagiarismRequest,
    background_tasks: BackgroundTasks
):
    """
    Detect plagiarism by comparing target text with candidate texts
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
        background_tasks: FastAPI background tasks for webhook calls
    
    Returns:
        PlagiarismResponse with similarity scores and plagiarism analysis
    """
    try:
        logger.info(f"Processing plagiarism detection request for target text length: {len(request.target_text)}")
        
        # Validate input
        if not request.target_text.strip():
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        
        if not request.candidate_texts:
            raise HTTPException(status_code=400, detail="Candidate texts array cannot be empty")
        
        # Process plagiarism detection
        result = await plagiarism_detector.detect_plagiarism(
            target_text=request.target_text,
            candidate_texts=request.candidate_texts,
            threshold=request.threshold,
            include_preprocessing=request.include_preprocessing
        )
        
        # Add webhook call to background tasks if webhook URL is provided
        if request.webhook_url:
            background_tasks.add_task(
                webhook_service.send_webhook,
                webhook_url=request.webhook_url,
                data=result.dict(),
                webhook_secret=request.webhook_secret
            )
            logger.info(f"Webhook scheduled for URL: {request.webhook_url}")
        
        logger.info(f"Plagiarism detection completed. Potential plagiarism detected: {result.has_plagiarism}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in plagiarism detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch-similarity", response_model=List[SimilarityResult])
async def batch_similarity(
    request: PlagiarismRequest
):
    """
    Calculate similarity scores without plagiarism analysis
    Useful for getting raw similarity metrics
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
    
    Returns:
        List of SimilarityResult with detailed similarity metrics
    """
    try:
        logger.info(f"Processing batch similarity request for {len(request.candidate_texts)} candidates")
        
        # Validate input
        if not request.target_text.strip():
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        
        if not request.candidate_texts:
            raise HTTPException(status_code=400, detail="Candidate texts array cannot be empty")
        
        # Calculate similarities
        similarities = await plagiarism_detector.calculate_similarities(
            target_text=request.target_text,
            candidate_texts=request.candidate_texts,
            include_preprocessing=request.include_preprocessing
        )
        
        logger.info(f"Batch similarity calculation completed for {len(similarities)} texts")
        return similarities
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch similarity calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/configure-webhook")
async def configure_webhook(webhook_config: WebhookConfig):
    """
    Configure default webhook settings for the API
    
    Args:
        webhook_config: WebhookConfig with webhook URL and optional secret
    
    Returns:
        Configuration confirmation
    """
    try:
        # Test webhook connectivity
        test_successful = await webhook_service.test_webhook(
            webhook_url=webhook_config.webhook_url,
            webhook_secret=webhook_config.webhook_secret
        )
        
        if test_successful:
            # Store webhook configuration (in production, this would be in a database)
            webhook_service.set_default_config(webhook_config)
            return {
                "status": "success",
                "message": "Webhook configured successfully",
                "webhook_url": webhook_config.webhook_url
            }
        else:
            raise HTTPException(status_code=400, detail="Webhook URL is not accessible")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure webhook: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
