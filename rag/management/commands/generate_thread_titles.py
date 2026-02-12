from django.core.management.base import BaseCommand
from rag.models import Thread, Message
from rag.utils import generate_thread_title
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Generate titles for existing threads that don\'t have titles yet'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without actually updating the database',
        )
        parser.add_argument(
            '--thread-id',
            type=str,
            help='Generate title for a specific thread ID only',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=10,
            help='Number of threads to process in each batch (default: 10)',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force regeneration of titles even for threads that already have titles',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        specific_thread_id = options.get('thread_id')
        batch_size = options['batch_size']
        force = options['force']
        
        self.stdout.write(
            self.style.SUCCESS('Starting thread title generation...')
        )
        
        # Get threads that need titles
        if specific_thread_id:
            if force:
                threads = Thread.objects.filter(id=specific_thread_id)
            else:
                threads = Thread.objects.filter(id=specific_thread_id, title__isnull=True)
            
            if not threads.exists():
                self.stdout.write(
                    self.style.WARNING(f'Thread {specific_thread_id} not found or already has a title.')
                )
                return
        else:
            if force:
                threads = Thread.objects.all().order_by('created_at')
            else:
                threads = Thread.objects.filter(title__isnull=True).order_by('created_at')
        
        total_threads = threads.count()
        if total_threads == 0:
            self.stdout.write(
                self.style.SUCCESS('No threads found that need titles.')
            )
            return
            
        self.stdout.write(f'Found {total_threads} threads that need titles.')
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made to the database')
            )
        
        processed = 0
        successful = 0
        failed = 0
        
        # Process threads in batches
        for i in range(0, total_threads, batch_size):
            batch = threads[i:i + batch_size]
            
            for thread in batch:
                try:
                    # Get the first user message for this thread
                    first_user_message = thread.messages.filter(role='user').order_by('created_at').first()
                    
                    if not first_user_message:
                        self.stdout.write(
                            self.style.WARNING(f'Thread {thread.id}: No user messages found, skipping.')
                        )
                        failed += 1
                        continue
                    
                    # Generate title
                    if dry_run:
                        title = generate_thread_title(first_user_message.content, thread.user)
                        current_title = thread.title or "No title"
                        self.stdout.write(
                            f'[DRY RUN] Thread {thread.id}: Current="{current_title}" -> New="{title}"'
                        )
                    else:
                        title = generate_thread_title(first_user_message.content, thread.user)
                        old_title = thread.title
                        thread.title = title
                        thread.save(update_fields=['title'])
                        
                        if old_title:
                            self.stdout.write(
                                self.style.SUCCESS(f'Thread {thread.id}: Updated title from "{old_title}" to "{title}"')
                            )
                        else:
                            self.stdout.write(
                                self.style.SUCCESS(f'Thread {thread.id}: Generated title "{title}"')
                            )
                    
                    successful += 1
                    
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Thread {thread.id}: Failed to generate title - {str(e)}')
                    )
                    failed += 1
                    
                    # Set a fallback title if not in dry run mode
                    if not dry_run:
                        try:
                            old_title = thread.title
                            thread.title = "New Conversation"
                            thread.save(update_fields=['title'])
                            self.stdout.write(
                                self.style.WARNING(f'Thread {thread.id}: Set fallback title "New Conversation" (was: "{old_title or "No title"}")')
                            )
                        except Exception as fallback_e:
                            self.stdout.write(
                                self.style.ERROR(f'Thread {thread.id}: Failed to set fallback title - {str(fallback_e)}')
                            )
                
                processed += 1
                
                # Progress update
                if processed % 10 == 0:
                    self.stdout.write(f'Processed {processed}/{total_threads} threads...')
        
        # Final summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write(f'Title generation completed!')
        self.stdout.write(f'Total processed: {processed}')
        self.stdout.write(f'Successful: {successful}')
        self.stdout.write(f'Failed: {failed}')
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('This was a dry run. Run without --dry-run to actually update the database.')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS('All thread titles have been generated!')
            )


# # First, run a dry run to see what would happen
# python manage.py generate_thread_titles --dry-run

# # Generate titles for all threads that don't have titles
# python manage.py generate_thread_titles

# # Generate title for a specific thread
# python manage.py generate_thread_titles --thread-id thread_12345

# # Force regenerate all titles (even existing ones)
# python manage.py generate_thread_titles --force

# # Process in smaller batches (useful for large datasets)
# python manage.py generate_thread_titles --batch-size 5