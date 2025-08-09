"""
Celery configuration for asynchronous task processing
"""

from celery import Celery
from config.settings import Settings

# Load settings
settings = Settings()
celery_config = settings.get_celery_config()

# Create Celery instance
celery_app = Celery(
    "plagiarism_detection",
    broker=celery_config["broker_url"],
    backend=celery_config["result_backend"],
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
    
    # Worker settings from configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    task_time_limit=celery_config["task_time_limit"],
    
    # Result backend settings from configuration
    result_expires=celery_config["result_expires"],
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

if __name__ == "__main__":
    celery_app.start()