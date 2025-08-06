"""
Webhook service for sending plagiarism detection results
"""

import aiohttp
import asyncio
import json
import logging
import hmac
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

from schemas.request_models import WebhookConfig

logger = logging.getLogger(__name__)

class WebhookService:
    """
    Service for handling webhook notifications
    Sends plagiarism detection results to configured webhook endpoints
    """
    
    def __init__(self):
        self.default_config: Optional[WebhookConfig] = None
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def send_webhook(
        self,
        webhook_url: str,
        data: Dict[str, Any],
        webhook_secret: Optional[str] = None,
        max_retries: int = 3
    ) -> bool:
        """
        Send webhook notification with plagiarism detection results
        
        Args:
            webhook_url: URL to send the webhook to
            data: Plagiarism detection results data
            webhook_secret: Optional secret for webhook authentication
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if webhook was sent successfully, False otherwise
        """
        try:
            # Prepare webhook payload
            payload = {
                "event": "plagiarism_detection_completed",
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Bangla-Plagiarism-Detector/1.0"
            }
            
            # Add signature if secret is provided
            if webhook_secret:
                signature = self._generate_signature(json.dumps(payload, default=str), webhook_secret)
                headers["X-Webhook-Signature"] = signature
            
            # Send webhook with retries
            session = await self._get_session()
            
            for attempt in range(max_retries + 1):
                try:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook sent successfully to {webhook_url}")
                            return True
                        elif response.status in [201, 202, 204]:
                            logger.info(f"Webhook accepted by {webhook_url} with status {response.status}")
                            return True
                        else:
                            logger.warning(f"Webhook failed with status {response.status}: {await response.text()}")
                            
                except aiohttp.ClientError as e:
                    logger.warning(f"Webhook attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < max_retries:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying webhook in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Webhook failed after {max_retries + 1} attempts")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error sending webhook: {str(e)}")
            return False
    
    async def test_webhook(
        self,
        webhook_url: str,
        webhook_secret: Optional[str] = None
    ) -> bool:
        """
        Test webhook connectivity by sending a test payload
        
        Args:
            webhook_url: URL to test
            webhook_secret: Optional secret for authentication
            
        Returns:
            True if webhook is accessible, False otherwise
        """
        try:
            test_payload = {
                "event": "webhook_test",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "This is a test webhook from Bangla Plagiarism Detection API"
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Bangla-Plagiarism-Detector/1.0"
            }
            
            if webhook_secret:
                signature = self._generate_signature(json.dumps(test_payload, default=str), webhook_secret)
                headers["X-Webhook-Signature"] = signature
            
            session = await self._get_session()
            
            async with session.post(
                webhook_url,
                json=test_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 201, 202, 204]:
                    logger.info(f"Webhook test successful for {webhook_url}")
                    return True
                else:
                    logger.warning(f"Webhook test failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Webhook test failed: {str(e)}")
            return False
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC signature for webhook authentication
        
        Args:
            payload: JSON payload as string
            secret: Webhook secret
            
        Returns:
            HMAC signature in hex format
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def set_default_config(self, config: WebhookConfig):
        """
        Set default webhook configuration
        
        Args:
            config: WebhookConfig object with default settings
        """
        self.default_config = config
        logger.info(f"Default webhook configuration set for {config.webhook_url}")
    
    def get_default_config(self) -> Optional[WebhookConfig]:
        """
        Get default webhook configuration
        
        Returns:
            Default WebhookConfig or None if not set
        """
        return self.default_config
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Webhook service HTTP session closed")
