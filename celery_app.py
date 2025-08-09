"""
Celery configuration for asynchronous task processing
"""

from celery import Celery
import os

# Configure Redis URL
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery instance
celery_app = Celery(
    "plagiarism_detection",
    broker=redis_url,
    backend=redis_url,
    include=["tasks.plagiarism_tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "tasks.plagiarism_tasks.process_plagiarism_detection": {"queue": "plagiarism"},
        "tasks.plagiarism_tasks.process_batch_similarity": {"queue": "similarity"}
    },
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

if __name__ == "__main__":
    celery_app.start()