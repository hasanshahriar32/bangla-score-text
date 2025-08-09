"""
Configuration settings for the plagiarism detection API
"""

import os
from typing import Optional

class Settings:
    """
    Configuration settings loaded from environment variables
    """
    
    def __init__(self):
        # API Configuration
        self.api_title: str = os.getenv("API_TITLE", "Bangla Plagiarism Detection API")
        self.api_version: str = os.getenv("API_VERSION", "1.0.0")
        self.api_description: str = os.getenv(
            "API_DESCRIPTION", 
            "ML-powered plagiarism detection system for Bangla language text with similarity scoring"
        )
        
        # Server Configuration
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", 5000))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # ML Model Configuration
        self.model_name: str = os.getenv(
            "MODEL_NAME", 
            "tfidf-bangla"
        )
        self.model_cache_dir: Optional[str] = os.getenv("MODEL_CACHE_DIR")
        
        # Plagiarism Detection Configuration
        self.default_threshold: float = float(os.getenv("DEFAULT_THRESHOLD", "0.7"))
        self.max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
        self.max_candidate_texts: int = int(os.getenv("MAX_CANDIDATE_TEXTS", "100"))
        
        # Webhook Configuration
        self.webhook_timeout: int = int(os.getenv("WEBHOOK_TIMEOUT", "30"))
        self.webhook_max_retries: int = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))
        self.webhook_secret: Optional[str] = os.getenv("WEBHOOK_SECRET")
        
        # Logging Configuration
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_format: str = os.getenv(
            "LOG_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Performance Configuration
        self.enable_preprocessing: bool = os.getenv("ENABLE_PREPROCESSING", "true").lower() == "true"
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
        
        # Redis & Queue Configuration
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db: int = int(os.getenv("REDIS_DB", "0"))
        self.redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
        
        # Celery Configuration
        self.celery_broker_url: str = os.getenv("CELERY_BROKER_URL", self.redis_url)
        self.celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", self.redis_url)
        self.celery_worker_concurrency: int = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))
        self.celery_task_time_limit: int = int(os.getenv("CELERY_TASK_TIME_LIMIT", "300"))  # 5 minutes
        self.celery_result_expires: int = int(os.getenv("CELERY_RESULT_EXPIRES", "3600"))  # 1 hour
        
        # Flower Monitoring Configuration
        self.flower_port: int = int(os.getenv("FLOWER_PORT", "5555"))
        self.flower_address: str = os.getenv("FLOWER_ADDRESS", "0.0.0.0")
        self.flower_basic_auth: Optional[str] = os.getenv("FLOWER_BASIC_AUTH")  # user:password format
        
        # Security Configuration
        self.cors_origins: list = os.getenv("CORS_ORIGINS", "*").split(",")
        self.api_key: Optional[str] = os.getenv("API_KEY")
        
        # Rate Limiting (for future implementation)
        self.rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
    
    def get_model_config(self) -> dict:
        """Get ML model configuration"""
        config = {
            "model_name": self.model_name,
            "cache_folder": self.model_cache_dir
        }
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
    
    def get_webhook_config(self) -> dict:
        """Get webhook configuration"""
        return {
            "timeout": self.webhook_timeout,
            "max_retries": self.webhook_max_retries,
            "default_secret": self.webhook_secret
        }
    
    def get_api_config(self) -> dict:
        """Get API configuration"""
        return {
            "title": self.api_title,
            "version": self.api_version,
            "description": self.api_description,
            "debug": self.debug
        }
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration prioritizing URL over individual parameters"""
        # Always prioritize the complete Redis URL if available
        if self.redis_url and self.redis_url != "redis://localhost:6379/0":
            return {"url": self.redis_url}
        
        # Fallback to individual parameters only if URL is default localhost
        config = {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db
        }
        
        if self.redis_password:
            config["password"] = self.redis_password
            
        return config
    
    def get_celery_config(self) -> dict:
        """Get Celery configuration"""
        return {
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend,
            "worker_concurrency": self.celery_worker_concurrency,
            "task_time_limit": self.celery_task_time_limit,
            "result_expires": self.celery_result_expires
        }
    
    def get_flower_config(self) -> dict:
        """Get Flower monitoring configuration"""
        config = {
            "port": self.flower_port,
            "address": self.flower_address
        }
        
        if self.flower_basic_auth:
            config["basic_auth"] = self.flower_basic_auth
            
        return config

    def get_plagiarism_config(self) -> dict:
        """Get plagiarism detection configuration"""
        return {
            "default_threshold": self.default_threshold,
            "max_text_length": self.max_text_length,
            "max_candidate_texts": self.max_candidate_texts,
            "enable_preprocessing": self.enable_preprocessing,
            "batch_size": self.batch_size
        }
    
    def validate_settings(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            True if all settings are valid, False otherwise
        """
        try:
            # Validate threshold
            if not 0.0 <= self.default_threshold <= 1.0:
                raise ValueError(f"Invalid threshold: {self.default_threshold}")
            
            # Validate port
            if not 1 <= self.port <= 65535:
                raise ValueError(f"Invalid port: {self.port}")
            
            # Validate timeouts
            if self.webhook_timeout < 1:
                raise ValueError(f"Invalid webhook timeout: {self.webhook_timeout}")
            
            # Validate limits
            if self.max_text_length < 1:
                raise ValueError(f"Invalid max text length: {self.max_text_length}")
            
            if self.max_candidate_texts < 1:
                raise ValueError(f"Invalid max candidate texts: {self.max_candidate_texts}")
            
            return True
            
        except ValueError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def __str__(self) -> str:
        """String representation of settings (excluding sensitive data)"""
        return f"""
Settings:
  API: {self.api_title} v{self.api_version}
  Host: {self.host}:{self.port}
  Model: {self.model_name}
  Threshold: {self.default_threshold}
  Debug: {self.debug}
  Preprocessing: {self.enable_preprocessing}
  Max Text Length: {self.max_text_length}
  Max Candidates: {self.max_candidate_texts}
"""
