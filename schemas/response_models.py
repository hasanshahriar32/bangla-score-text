"""
Pydantic models for API responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class SimilarityResult(BaseModel):
    """Individual similarity result for a candidate text"""
    
    candidate_index: int = Field(
        ...,
        description="Index of the candidate text in the original array"
    )
    
    candidate_text: str = Field(
        ...,
        description="The candidate text that was compared"
    )
    
    similarity_score: float = Field(
        ...,
        description="Similarity score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score for the similarity calculation",
        ge=0.0,
        le=1.0
    )
    
    length_ratio: float = Field(
        ...,
        description="Ratio of candidate text length to target text length"
    )
    
    target_length: int = Field(
        ...,
        description="Length of the target text in characters"
    )
    
    candidate_length: int = Field(
        ...,
        description="Length of the candidate text in characters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidate_index": 0,
                "candidate_text": "এটি একটি নমুনা টেক্সট।",
                "similarity_score": 0.85,
                "confidence": 0.82,
                "length_ratio": 0.7,
                "target_length": 50,
                "candidate_length": 35
            }
        }

class PlagiarismAnalysis(BaseModel):
    """Detailed plagiarism analysis results"""
    
    potential_plagiarism: bool = Field(
        ...,
        description="Whether potential plagiarism was detected"
    )
    
    matches_above_threshold: int = Field(
        ...,
        description="Number of candidate texts with similarity above threshold"
    )
    
    max_similarity_score: float = Field(
        ...,
        description="Maximum similarity score found",
        ge=0.0,
        le=1.0
    )
    
    average_similarity_score: float = Field(
        ...,
        description="Average similarity score across all candidates",
        ge=0.0,
        le=1.0
    )
    
    risk_level: str = Field(
        ...,
        description="Risk level: low, medium, high, or critical"
    )
    
    matched_candidate_indices: List[int] = Field(
        ...,
        description="Indices of candidate texts that matched above threshold"
    )
    
    analysis_summary: str = Field(
        ...,
        description="Human-readable summary of the analysis"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "potential_plagiarism": True,
                "matches_above_threshold": 2,
                "max_similarity_score": 0.89,
                "average_similarity_score": 0.65,
                "risk_level": "high",
                "matched_candidate_indices": [0, 2],
                "analysis_summary": "Potential plagiarism detected with 2 matches above threshold. Risk level: HIGH. Maximum similarity score: 0.890, Average similarity: 0.650. HIGH RISK: Significant similarity detected. Manual review recommended."
            }
        }

class PlagiarismResponse(BaseModel):
    """Complete plagiarism detection response"""
    
    target_text: str = Field(
        ...,
        description="The original target text that was analyzed"
    )
    
    similarity_results: List[SimilarityResult] = Field(
        ...,
        description="Detailed similarity results for each candidate text"
    )
    
    plagiarism_analysis: PlagiarismAnalysis = Field(
        ...,
        description="Comprehensive plagiarism analysis"
    )
    
    threshold_used: float = Field(
        ...,
        description="Similarity threshold that was used for analysis",
        ge=0.0,
        le=1.0
    )
    
    has_plagiarism: bool = Field(
        ...,
        description="Quick flag indicating if plagiarism was detected"
    )
    
    max_similarity_score: float = Field(
        ...,
        description="Maximum similarity score found",
        ge=0.0,
        le=1.0
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp when the analysis was performed"
    )
    
    preprocessing_applied: bool = Field(
        ...,
        description="Whether text preprocessing was applied"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_text": "এটি একটি নমুনা বাংলা টেক্সট যা পরীক্ষার জন্য ব্যবহৃত হচ্ছে।",
                "similarity_results": [
                    {
                        "candidate_index": 0,
                        "candidate_text": "এটি একটি নমুনা বাংলা টেক্সট।",
                        "similarity_score": 0.89,
                        "confidence": 0.85,
                        "length_ratio": 0.7,
                        "target_length": 50,
                        "candidate_length": 35
                    }
                ],
                "plagiarism_analysis": {
                    "potential_plagiarism": True,
                    "matches_above_threshold": 1,
                    "max_similarity_score": 0.89,
                    "average_similarity_score": 0.65,
                    "risk_level": "high",
                    "matched_candidate_indices": [0],
                    "analysis_summary": "Potential plagiarism detected with 1 matches above threshold. Risk level: HIGH."
                },
                "threshold_used": 0.7,
                "has_plagiarism": True,
                "max_similarity_score": 0.89,
                "timestamp": "2025-08-06T10:30:00.000Z",
                "preprocessing_applied": True
            }
        }

class QueuedTaskResponse(BaseModel):
    """Response for queued task submission"""
    
    task_id: str = Field(
        ...,
        description="Unique task identifier"
    )
    
    status: str = Field(
        ...,
        description="Task status"
    )
    
    message: str = Field(
        ...,
        description="Status message"
    )
    
    estimated_completion_time: Optional[str] = Field(
        None,
        description="Estimated completion time in seconds"
    )
    
    queue_position: Optional[int] = Field(
        None,
        description="Position in the processing queue"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "queued",
                "message": "Task queued successfully for processing",
                "estimated_completion_time": "30-60 seconds",
                "queue_position": 2
            }
        }

class TaskStatusResponse(BaseModel):
    """Response for task status check"""
    
    task_id: str = Field(
        ...,
        description="Task identifier"
    )
    
    status: str = Field(
        ...,
        description="Current task status (PENDING, PROCESSING, SUCCESS, FAILURE)"
    )
    
    progress: Optional[int] = Field(
        None,
        description="Task progress percentage (0-100)"
    )
    
    message: Optional[str] = Field(
        None,
        description="Current status message"
    )
    
    result: Optional[dict] = Field(
        None,
        description="Task result (only available when completed)"
    )
    
    started_at: Optional[str] = Field(
        None,
        description="Task start timestamp"
    )
    
    completed_at: Optional[str] = Field(
        None,
        description="Task completion timestamp"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message (only if failed)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "abc123-def456-ghi789",
                "status": "PROCESSING",
                "progress": 45,
                "message": "ML model loaded, processing texts",
                "started_at": "2025-08-06T16:52:17.057029"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(
        ...,
        description="API status"
    )
    
    message: str = Field(
        ...,
        description="Status message"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    redis_status: Optional[str] = Field(
        None,
        description="Redis connection status"
    )
    
    celery_status: Optional[str] = Field(
        None,
        description="Celery worker status"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Bangla Plagiarism Detection API is running",
                "version": "1.0.0",
                "redis_status": "connected",
                "celery_status": "active"
            }
        }
