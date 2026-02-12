from django.apps import AppConfig
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag'

    def ready(self):
        """Called when Django starts up"""
        self.validate_api_keys()

    def validate_api_keys(self):
        """Validate that required API keys are configured"""
        try:
            # Check OpenAI API Key
            openai_key = getattr(settings, 'OPENAI_API_KEY', None)
            if not openai_key or openai_key.strip() == '':
                logger.error("=" * 60)
                logger.error("❌ CRITICAL ERROR: OPENAI_API_KEY is missing or empty!")
                logger.error("   Please set OPENAI_API_KEY in your settings.py or environment variables.")
                logger.error("   Example: OPENAI_API_KEY = 'sk-your-api-key-here'")
                logger.error("=" * 60)
                raise ValueError("OPENAI_API_KEY is required but not configured")
            
            # Validate the key format (basic check)
            if not openai_key.startswith('sk-'):
                logger.warning("⚠️  WARNING: OPENAI_API_KEY doesn't start with 'sk-' - this might be invalid")
            
            logger.info("✅ OPENAI_API_KEY is configured")
            
            # Check other required settings
            self.validate_other_settings()
            
        except Exception as e:
            logger.critical(f" STARTUP VALIDATION FAILED: {e}")
            logger.critical("Server cannot start without proper API key configuration.")
            raise

    def validate_other_settings(self):
        """Validate other critical settings"""
        # Check Qdrant settings
        qdrant_host = getattr(settings, 'QDRANT_HOST', None)
        qdrant_port = getattr(settings, 'QDRANT_PORT', None)
        
        if not qdrant_host or not qdrant_port:
            logger.warning("⚠️  WARNING: QDRANT_HOST or QDRANT_PORT not configured")
        else:
            logger.info(f"✅ Qdrant configured: {qdrant_host}:{qdrant_port}")
        
        # Check default models
        default_model = getattr(settings, 'DEFAULT_OPENAI_MODEL', None)
        if default_model:
            logger.info(f"✅ Default OpenAI model: {default_model}")
        
        logger.info("🎉 All startup validations passed!")
