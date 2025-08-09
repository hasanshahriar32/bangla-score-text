"""
FastAPI-based Plagiarism Detection System for Bangla Language
Provides text similarity scoring and plagiarism detection with webhook integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from typing import List
import logging
import os

from schemas.request_models import PlagiarismRequest, WebhookConfig
from schemas.response_models import (
    PlagiarismResponse, SimilarityResult, HealthResponse, 
    QueuedTaskResponse, TaskStatusResponse
)
from models.plagiarism_detector import PlagiarismDetector
from services.webhook_service import WebhookService
from config.settings import Settings
from celery_app import celery_app
from tasks.plagiarism_tasks import process_plagiarism_detection, process_batch_similarity
import redis

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

# Redis connection for health checks
redis_client = None

def get_redis_client():
    """Get Redis client instance"""
    global redis_client
    if redis_client is None:
        try:
            redis_config = settings.get_redis_config()
            redis_client = redis.from_url(redis_config["url"])
            redis_client.ping()  # Test connection
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    return redis_client

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
        version="1.0.0",
        redis_status="unknown",
        celery_status="unknown"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with model status"""
    model_status = "loaded" if plagiarism_detector.is_initialized else "not_loaded"
    
    # Check Redis connection
    redis_status = "disconnected"
    try:
        client = get_redis_client()
        if client:
            client.ping()
            redis_status = "connected"
    except Exception:
        redis_status = "disconnected"
    
    # Check Celery worker status
    celery_status = "unknown"
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        celery_status = "active" if active_workers else "inactive"
    except Exception:
        celery_status = "unavailable"
    
    return HealthResponse(
        status="healthy",
        message=f"API is running, ML model status: {model_status}",
        version="1.0.0",
        redis_status=redis_status,
        celery_status=celery_status
    )

@app.get("/flower")
async def flower_dashboard():
    """Provide access to Flower monitoring dashboard"""
    # Return an HTML page that redirects to Flower or embeds it
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flower Dashboard - Queue Monitor</title>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
            .header { background: #f5f5f5; padding: 10px; margin-bottom: 20px; border-radius: 5px; }
            iframe { width: 100%; height: 80vh; border: 1px solid #ddd; border-radius: 5px; }
            .info { background: #e3f2fd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🌸 Flower Dashboard - Redis & Queue Monitor</h2>
            <p>Monitor Celery workers, Redis connections, and task queues</p>
        </div>
        <div class="info">
            <strong>Direct Access:</strong> <a href="http://localhost:5555" target="_blank">http://localhost:5555</a>
        </div>
        <iframe src="http://localhost:5555" frameborder="0"></iframe>
        <script>
            // Refresh iframe every 30 seconds to keep data fresh
            setInterval(function() {
                document.querySelector('iframe').src = document.querySelector('iframe').src;
            }, 30000);
        </script>
    </body>
    </html>
    """)

@app.post("/detect-plagiarism", response_model=QueuedTaskResponse)
async def queue_plagiarism_detection(request: PlagiarismRequest):
    """
    Queue plagiarism detection task for processing
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
    
    Returns:
        QueuedTaskResponse with task ID and status
    """
    try:
        logger.info(f"Queueing plagiarism detection for target text length: {len(request.target_text)}")
        
        # Validate input
        if not request.target_text.strip():
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        
        if not request.candidate_texts:
            raise HTTPException(status_code=400, detail="Candidate texts array cannot be empty")
        
        # Prepare task data
        task_data = {
            "request_data": request.dict(exclude={"webhook_url", "webhook_secret"}),
            "webhook_url": request.webhook_url,
            "webhook_secret": request.webhook_secret
        }
        
        # Queue the task
        task = process_plagiarism_detection.apply_async(args=[task_data])
        
        logger.info(f"Plagiarism detection task queued with ID: {task.id}")
        
        return QueuedTaskResponse(
            task_id=task.id,
            status="queued",
            message="Plagiarism detection task queued successfully. Check task status or wait for webhook notification.",
            estimated_completion_time="30-120 seconds"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing plagiarism detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")

@app.post("/detect-plagiarism-sync", response_model=PlagiarismResponse)
async def detect_plagiarism_sync(
    request: PlagiarismRequest,
    background_tasks: BackgroundTasks
):
    """
    Detect plagiarism synchronously (for smaller texts, legacy endpoint)
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
        background_tasks: FastAPI background tasks for webhook calls
    
    Returns:
        PlagiarismResponse with similarity scores and plagiarism analysis
    """
    try:
        logger.info(f"Processing synchronous plagiarism detection for target text length: {len(request.target_text)}")
        
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
            include_preprocessing=request.include_preprocessing or True
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

@app.post("/batch-similarity", response_model=QueuedTaskResponse)
async def queue_batch_similarity(request: PlagiarismRequest):
    """
    Queue batch similarity calculation task
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
    
    Returns:
        QueuedTaskResponse with task ID and status
    """
    try:
        logger.info(f"Queueing batch similarity for {len(request.candidate_texts)} candidates")
        
        # Validate input
        if not request.target_text.strip():
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        
        if not request.candidate_texts:
            raise HTTPException(status_code=400, detail="Candidate texts array cannot be empty")
        
        # Prepare task data
        task_data = {
            "request_data": request.dict(exclude={"webhook_url", "webhook_secret"}),
            "webhook_url": request.webhook_url,
            "webhook_secret": request.webhook_secret
        }
        
        # Queue the task
        task = process_batch_similarity.apply_async(args=[task_data])
        
        logger.info(f"Batch similarity task queued with ID: {task.id}")
        
        return QueuedTaskResponse(
            task_id=task.id,
            status="queued", 
            message="Batch similarity task queued successfully. Check task status or wait for webhook notification.",
            estimated_completion_time="15-60 seconds"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queueing batch similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")

@app.post("/batch-similarity-sync", response_model=List[SimilarityResult])
async def batch_similarity_sync(request: PlagiarismRequest):
    """
    Calculate similarity scores synchronously (for smaller datasets, legacy endpoint)
    
    Args:
        request: PlagiarismRequest containing target text and candidate texts
    
    Returns:
        List of SimilarityResult with detailed similarity metrics
    """
    try:
        logger.info(f"Processing synchronous batch similarity for {len(request.candidate_texts)} candidates")
        
        # Validate input
        if not request.target_text.strip():
            raise HTTPException(status_code=400, detail="Target text cannot be empty")
        
        if not request.candidate_texts:
            raise HTTPException(status_code=400, detail="Candidate texts array cannot be empty")
        
        # Calculate similarities
        similarities = await plagiarism_detector.calculate_similarities(
            target_text=request.target_text,
            candidate_texts=request.candidate_texts,
            include_preprocessing=request.include_preprocessing or True
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

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a queued task
    
    Args:
        task_id: The task identifier
        
    Returns:
        TaskStatusResponse with current task status and progress
    """
    try:
        # Get task result from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == "PENDING":
            return TaskStatusResponse(
                task_id=task_id,
                status="PENDING",
                message="Task is waiting to be processed"
            )
        elif task_result.state == "PROCESSING":
            meta = task_result.info or {}
            return TaskStatusResponse(
                task_id=task_id,
                status="PROCESSING",
                progress=meta.get("progress", 0),
                message=meta.get("status", "Processing"),
                started_at=meta.get("started_at")
            )
        elif task_result.state == "SUCCESS":
            meta = task_result.info or {}
            return TaskStatusResponse(
                task_id=task_id,
                status="SUCCESS",
                progress=100,
                message="Task completed successfully",
                result=meta.get("result"),
                started_at=meta.get("started_at"),
                completed_at=meta.get("completed_at")
            )
        elif task_result.state == "FAILURE":
            meta = task_result.info or {}
            return TaskStatusResponse(
                task_id=task_id,
                status="FAILURE",
                message="Task failed",
                error=meta.get("error", str(task_result.info)),
                started_at=meta.get("started_at"),
                completed_at=meta.get("failed_at")
            )
        else:
            return TaskStatusResponse(
                task_id=task_id,
                status=task_result.state,
                message=f"Task status: {task_result.state}"
            )
            
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/queue/stats")
async def get_queue_stats():
    """
    Get queue statistics and worker information
    
    Returns:
        Dictionary with queue and worker statistics
    """
    try:
        inspect = celery_app.control.inspect()
        
        # Get active tasks
        active_tasks = inspect.active() or {}
        
        # Get worker stats
        stats = inspect.stats() or {}
        
        # Count total active tasks
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        
        # Get queue lengths (requires Redis connection)
        queue_lengths = {}
        try:
            client = get_redis_client()
            if client:
                queue_lengths = {
                    "plagiarism": client.llen("plagiarism"),
                    "similarity": client.llen("similarity"),
                    "default": client.llen("celery")
                }
        except Exception:
            queue_lengths = {"error": "Unable to connect to Redis"}
        
        return {
            "active_tasks": total_active,
            "workers": list(stats.keys()),
            "worker_count": len(stats),
            "queue_lengths": queue_lengths,
            "active_tasks_by_worker": active_tasks,
            "worker_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {str(e)}")
        return {
            "error": f"Failed to get queue stats: {str(e)}",
            "active_tasks": 0,
            "workers": [],
            "worker_count": 0
        }

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
