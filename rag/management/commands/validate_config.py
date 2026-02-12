from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os

class Command(BaseCommand):
    help = 'Validate that all required API keys and configurations are set'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('�� Validating configuration...'))
        
        errors = []
        warnings = []
        
        # Check OpenAI API Key
        openai_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not openai_key or openai_key.strip() == '':
            errors.append("❌ OPENAI_API_KEY is missing or empty")
        elif not openai_key.startswith('sk-'):
            warnings.append("⚠️  OPENAI_API_KEY doesn't start with 'sk-' - might be invalid")
        else:
            self.stdout.write(self.style.SUCCESS("✅ OPENAI_API_KEY is configured"))
        
        # Check Qdrant settings
        qdrant_host = getattr(settings, 'QDRANT_HOST', None)
        qdrant_port = getattr(settings, 'QDRANT_PORT', None)
        
        if not qdrant_host:
            errors.append("❌ QDRANT_HOST is not configured")
        elif not qdrant_port:
            errors.append("❌ QDRANT_PORT is not configured")
        else:
            self.stdout.write(self.style.SUCCESS(f"✅ Qdrant configured: {qdrant_host}:{qdrant_port}"))
        
        # Check database
        try:
            from django.db import connection
            connection.ensure_connection()
            self.stdout.write(self.style.SUCCESS("✅ Database connection successful"))
        except Exception as e:
            errors.append(f"❌ Database connection failed: {e}")
        
        # Display warnings
        for warning in warnings:
            self.stdout.write(self.style.WARNING(warning))
        
        # Display errors and exit if any
        if errors:
            self.stdout.write(self.style.ERROR("\n🚨 CONFIGURATION ERRORS:"))
            for error in errors:
                self.stdout.write(self.style.ERROR(error))
            
            self.stdout.write(self.style.ERROR("\n💡 To fix these issues:"))
            self.stdout.write(self.style.ERROR("1. Set OPENAI_API_KEY in your environment variables:"))
            self.stdout.write(self.style.ERROR("   export OPENAI_API_KEY='sk-your-api-key-here'"))
            self.stdout.write(self.style.ERROR("2. Or set it in settings.py:"))
            self.stdout.write(self.style.ERROR("   OPENAI_API_KEY = 'sk-your-api-key-here'"))
            
            raise CommandError("Configuration validation failed")
        
        self.stdout.write(self.style.SUCCESS("\n🎉 All configurations are valid!"))
