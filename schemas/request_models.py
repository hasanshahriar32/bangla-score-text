"""
Pydantic models for API request validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re

class PlagiarismRequest(BaseModel):
    """Request model for plagiarism detection API"""
    
    target_text: str = Field(
        ...,
        description="The text to check for plagiarism",
        min_length=1,
        max_length=50000
    )
    
    candidate_texts: List[str] = Field(
        ...,
        description="Array of texts to compare against the target text",
        min_items=1,
        max_items=100
    )
    
    threshold: Optional[float] = Field(
        0.7,
        description="Similarity threshold for plagiarism detection (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    include_preprocessing: Optional[bool] = Field(
        True,
        description="Whether to apply text preprocessing before analysis"
    )
    
    webhook_url: Optional[str] = Field(
        None,
        description="Optional webhook URL to send results to"
    )
    
    webhook_secret: Optional[str] = Field(
        None,
        description="Optional secret for webhook authentication"
    )
    
    @validator('target_text')
    def validate_target_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Target text cannot be empty or just whitespace')
        return v.strip()
    
    @validator('candidate_texts')
    def validate_candidate_texts(cls, v):
        if not v:
            raise ValueError('Candidate texts array cannot be empty')
        
        # Filter out empty texts and validate
        valid_texts = []
        for i, text in enumerate(v):
            if text and text.strip():
                if len(text) > 50000:
                    raise ValueError(f'Candidate text at index {i} exceeds maximum length of 50000 characters')
                valid_texts.append(text.strip())
            else:
                raise ValueError(f'Candidate text at index {i} cannot be empty or just whitespace')
        
        if not valid_texts:
            raise ValueError('No valid candidate texts provided')
        
        return valid_texts
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        if v is not None:
            # Basic URL validation
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            
            if not url_pattern.match(v):
                raise ValueError('Invalid webhook URL format')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "target_text": "এটি একটি নমুনা বাংলা টেক্সট যা পরীক্ষার জন্য ব্যবহৃত হচ্ছে।",
                "candidate_texts": [
                    "এটি একটি নমুনা বাংলা টেক্সট।",
                    "এটি সম্পূর্ণ ভিন্ন একটি টেক্সট।",
                    "একটি নমুনা বাংলা টেক্সট যা পরীক্ষার জন্য ব্যবহৃত।"
                ],
                "threshold": 0.7,
                "include_preprocessing": True,
                "webhook_url": "https://example.com/webhook",
                "webhook_secret": "optional_secret_key"
            }
        }

class WebhookConfig(BaseModel):
    """Configuration model for webhook settings"""
    
    webhook_url: str = Field(
        ...,
        description="Webhook URL to send results to"
    )
    
    webhook_secret: Optional[str] = Field(
        None,
        description="Optional secret for webhook authentication"
    )
    
    timeout: Optional[int] = Field(
        30,
        description="Webhook timeout in seconds",
        ge=1,
        le=300
    )
    
    retry_count: Optional[int] = Field(
        3,
        description="Number of retry attempts for failed webhooks",
        ge=0,
        le=10
    )
    
    @validator('webhook_url')
    def validate_webhook_url(cls, v):
        # Basic URL validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not url_pattern.match(v):
            raise ValueError('Invalid webhook URL format')
        
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "webhook_url": "https://example.com/webhook",
                "webhook_secret": "my_secret_key",
                "timeout": 30,
                "retry_count": 3
            }
        }
