"""
Celery tasks for plagiarism detection and batch similarity processing
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from celery_app import celery_app
from models.plagiarism_detector import PlagiarismDetector
from services.webhook_service import WebhookService
from schemas.request_models import PlagiarismRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services (will be done in each task to avoid pickle issues)
webhook_service = WebhookService()

@celery_app.task(bind=True)
def process_plagiarism_detection(self, task_data: Dict[str, Any]):
    """
    Celery task for processing plagiarism detection
    
    Args:
        task_data: Dictionary containing request data and metadata
    """
    try:
        # Extract task information
        request_data = task_data["request_data"]
        task_id = self.request.id
        webhook_url = task_data.get("webhook_url")
        webhook_secret = task_data.get("webhook_secret")
        
        logger.info(f"Starting plagiarism detection task {task_id}")
        print(f"🔍 Processing plagiarism detection task: {task_id}")
        print(f"📄 Target text length: {len(request_data['target_text'])}")
        print(f"📚 Number of candidates: {len(request_data['candidate_texts'])}")
        
        # Update task status
        self.update_state(
            state="PROCESSING",
            meta={
                "status": "Processing plagiarism detection",
                "progress": 10,
                "task_id": task_id,
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize plagiarism detector
        detector = PlagiarismDetector()
        
        # Run the async initialization in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize the detector
            loop.run_until_complete(detector.initialize())
            
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "ML model loaded, processing texts",
                    "progress": 30,
                    "task_id": task_id
                }
            )
            
            # Process the plagiarism detection
            result = loop.run_until_complete(
                detector.detect_plagiarism(
                    target_text=request_data["target_text"],
                    candidate_texts=request_data["candidate_texts"],
                    threshold=request_data.get("threshold", 0.7),
                    include_preprocessing=request_data.get("include_preprocessing", True)
                )
            )
            
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "Analysis complete, preparing results",
                    "progress": 80,
                    "task_id": task_id
                }
            )
            
            # Convert result to dict for serialization
            result_dict = result.dict()
            result_dict["task_id"] = task_id
            result_dict["completed_at"] = datetime.utcnow().isoformat()
            
            # Print results to console
            print(f"✅ Plagiarism detection completed for task {task_id}")
            print(f"🎯 Plagiarism detected: {result.has_plagiarism}")
            print(f"📊 Max similarity: {result.max_similarity_score:.3f}")
            print(f"🚨 Risk level: {result.plagiarism_analysis.risk_level.upper()}")
            print(f"🔢 Matches above threshold: {result.plagiarism_analysis.matches_above_threshold}")
            
            # Send webhook if URL provided
            if webhook_url:
                webhook_success = loop.run_until_complete(
                    webhook_service.send_webhook(
                        webhook_url=webhook_url,
                        data=result_dict,
                        webhook_secret=webhook_secret
                    )
                )
                if webhook_success:
                    print(f"📡 Webhook sent successfully for task {task_id}")
                else:
                    print(f"❌ Webhook failed for task {task_id}")
            
            # Final task completion
            self.update_state(
                state="SUCCESS",
                meta={
                    "status": "Completed successfully",
                    "progress": 100,
                    "task_id": task_id,
                    "result": result_dict
                }
            )
            
            return result_dict
            
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = f"Error in plagiarism detection task: {str(e)}"
        logger.error(error_msg)
        print(f"❌ Task {task_id} failed: {error_msg}")
        
        self.update_state(
            state="FAILURE",
            meta={
                "status": "Failed",
                "error": error_msg,
                "task_id": task_id,
                "failed_at": datetime.utcnow().isoformat()
            }
        )
        raise

@celery_app.task(bind=True)
def process_batch_similarity(self, task_data: Dict[str, Any]):
    """
    Celery task for processing batch similarity calculation
    
    Args:
        task_data: Dictionary containing request data and metadata
    """
    try:
        # Extract task information
        request_data = task_data["request_data"]
        task_id = self.request.id
        webhook_url = task_data.get("webhook_url")
        webhook_secret = task_data.get("webhook_secret")
        
        logger.info(f"Starting batch similarity task {task_id}")
        print(f"📊 Processing batch similarity task: {task_id}")
        print(f"📄 Target text length: {len(request_data['target_text'])}")
        print(f"📚 Number of candidates: {len(request_data['candidate_texts'])}")
        
        # Update task status
        self.update_state(
            state="PROCESSING",
            meta={
                "status": "Processing batch similarity",
                "progress": 10,
                "task_id": task_id,
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize plagiarism detector
        detector = PlagiarismDetector()
        
        # Run the async processing in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize the detector
            loop.run_until_complete(detector.initialize())
            
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "ML model loaded, calculating similarities",
                    "progress": 30,
                    "task_id": task_id
                }
            )
            
            # Process the batch similarity
            similarities = loop.run_until_complete(
                detector.calculate_similarities(
                    target_text=request_data["target_text"],
                    candidate_texts=request_data["candidate_texts"],
                    include_preprocessing=request_data.get("include_preprocessing", True)
                )
            )
            
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "status": "Similarities calculated, preparing results",
                    "progress": 80,
                    "task_id": task_id
                }
            )
            
            # Convert results to dict for serialization
            result_dict = {
                "task_id": task_id,
                "target_text": request_data["target_text"],
                "similarities": [sim.dict() for sim in similarities],
                "completed_at": datetime.utcnow().isoformat(),
                "total_candidates": len(similarities)
            }
            
            # Print results to console
            print(f"✅ Batch similarity completed for task {task_id}")
            print(f"📊 Processed {len(similarities)} candidates")
            max_similarity = max([sim.similarity_score for sim in similarities]) if similarities else 0
            print(f"🎯 Max similarity found: {max_similarity:.3f}")
            
            # Send webhook if URL provided
            if webhook_url:
                webhook_success = loop.run_until_complete(
                    webhook_service.send_webhook(
                        webhook_url=webhook_url,
                        data=result_dict,
                        webhook_secret=webhook_secret
                    )
                )
                if webhook_success:
                    print(f"📡 Webhook sent successfully for task {task_id}")
                else:
                    print(f"❌ Webhook failed for task {task_id}")
            
            # Final task completion
            self.update_state(
                state="SUCCESS",
                meta={
                    "status": "Completed successfully",
                    "progress": 100,
                    "task_id": task_id,
                    "result": result_dict
                }
            )
            
            return result_dict
            
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = f"Error in batch similarity task: {str(e)}"
        logger.error(error_msg)
        print(f"❌ Task {task_id} failed: {error_msg}")
        
        self.update_state(
            state="FAILURE",
            meta={
                "status": "Failed",
                "error": error_msg,
                "task_id": task_id,
                "failed_at": datetime.utcnow().isoformat()
            }
        )
        raise