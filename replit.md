# Bangla Plagiarism Detection API

## Overview

A FastAPI-based plagiarism detection system specifically designed for Bangla language text processing with advanced queue-based processing. The system uses multilingual machine learning models to calculate text similarity scores and detect potential plagiarism between a target text and multiple candidate texts. Features include configurable similarity thresholds, text preprocessing optimized for Bangla language, webhook integration for asynchronous result delivery, and a complete Celery-Redis queue system for handling longer processing tasks.

## User Preferences

Preferred communication style: Simple, everyday language.

## Production Deployment

### Environment Variables
The system requires these environment variables to be configured:

**Required:**
- `REDIS_URL` - Complete Redis connection string for queue and result storage
- `WEBHOOK_SECRET` - Secret key for securing webhook deliveries (optional)

**Optional Configuration:**
- `CELERY_WORKER_CONCURRENCY` - Number of concurrent workers (default: 2)
- `DEFAULT_THRESHOLD` - Default plagiarism threshold (default: 0.7)
- `MAX_TEXT_LENGTH` - Maximum text length limit (default: 50000)

### Queue System Architecture
- Redis serves as both message broker and result backend
- Celery handles distributed task processing
- Flower provides web-based queue monitoring
- Tasks are routed to specialized queues (plagiarism, similarity)

## System Architecture

### API Layer
- **Framework**: FastAPI with automatic OpenAPI documentation
- **CORS**: Configured to allow all origins for cross-platform access
- **Request Validation**: Pydantic models for strict input validation and type checking
- **Response Models**: Structured JSON responses with detailed similarity metrics

### Machine Learning Pipeline
- **Similarity Model**: TF-IDF vectorization with character n-grams (2-5) optimized for Bangla text
- **Text Processing**: Custom preprocessing pipeline optimized for Bangla Unicode text
- **Scoring Algorithm**: Cosine similarity calculation with confidence metrics
- **Language Support**: Character-based analysis supporting Bangla and multilingual text
- **Performance**: Lightweight implementation without heavy transformer dependencies

### Text Processing Engine
- **Bangla Optimization**: Unicode normalization and Bangla-specific text cleaning
- **Preprocessing Pipeline**: Handles text normalization, whitespace cleanup, and punctuation removal
- **Character Encoding**: UTF-8 support with proper Bangla Unicode range handling
- **Length Analysis**: Text length comparison and ratio calculations

### Configuration Management
- **Environment-based**: All settings configurable via environment variables
- **Model Configuration**: Customizable ML model selection and caching
- **Threshold Settings**: Adjustable plagiarism detection sensitivity
- **Performance Limits**: Configurable text length and candidate count limits

### Background Processing & Queue System
- **Celery Integration**: Distributed task queue using Redis as message broker
- **Redis Backend**: Fast in-memory result storage and queue management
- **Task Monitoring**: Flower web interface for real-time queue visualization
- **Queue Routing**: Dedicated queues for plagiarism detection and batch similarity
- **Progress Tracking**: Real-time task status updates with progress indicators
- **Webhook Service**: Asynchronous result delivery to external endpoints
- **Retry Mechanism**: Configurable retry logic for failed webhook deliveries
- **Authentication**: HMAC-based webhook security with secret validation
- **Timeout Handling**: HTTP client timeout configuration for external calls

### Service Architecture
The application follows a layered architecture pattern:
- **Controllers**: FastAPI route handlers for API endpoints
- **Services**: Business logic separation with dedicated webhook service
- **Models**: ML model abstraction with async initialization
- **Utilities**: Reusable text processing and validation components
- **Schemas**: Strict data validation using Pydantic models

## External Dependencies

### Machine Learning Framework
- **sentence-transformers**: Multilingual text embedding and similarity calculation
- **scikit-learn**: Cosine similarity computation and ML utilities
- **numpy**: Numerical operations and array processing

### Web Framework & Queue System
- **FastAPI**: Modern async web framework with automatic API documentation
- **uvicorn**: ASGI server for production deployment
- **aiohttp**: Async HTTP client for webhook delivery
- **Celery**: Distributed task queue for background processing
- **Redis**: Message broker and result backend for queue system
- **Flower**: Web-based monitoring tool for Celery tasks

### Data Validation
- **Pydantic**: Runtime type checking and data validation
- **typing**: Python type hints and annotations

### Text Processing
- **unicodedata**: Unicode normalization for Bangla text
- **re**: Regular expression processing for text cleaning

The system is designed to run as a standalone API service with no persistent database requirements, making it suitable for containerized deployment and microservice architectures.