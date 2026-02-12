import logging
from django.core.management.base import BaseCommand
from rag.models import Document, DocumentAlert
from rag.utils import extract_text_from_file
import os

logger = logging.getLogger(__name__) # Use Django's logger


class Command(BaseCommand):
    help = 'Scan existing documents and generate missing alerts'

    def handle(self, *args, **kwargs):
        documents = Document.objects.all()
        total_scanned = 0

        alert_keywords = [
                # Contract & Expiry
                "contract expiry", "contract end date", "renewal deadline", "service termination", "expiry notice",
                # Payments
                "payment due", "payment overdue", "invoice overdue", "late fee", "unpaid invoice", "outstanding balance", "collection notice",
                # Legal Risks
                "breach of contract", "penalty clause", "legal action", "non-compliance", "lawsuit", "settlement",
                # Deadlines
                "submission deadline", "due date", "project deadline", "final notice", "critical timeline",
                # Financial
                "advance payment", "refund request", "debit note", "credit note", "balance payable",
                # Risk Specific
                "termination for cause", "default notice", "breach penalty", "financial exposure",
                # Communication
                "no response received", "pending approval", "awaiting confirmation",
                # Supply Chain
                "shipment delay", "logistics issue", "supply disruption",
                # Tax / Regulatory
                "tax penalty", "compliance audit", "regulatory fine",
                # Partner/Vendor Risks
                "partner dispute", "vendor breach", "service level failure",

                # Additional keywords from the provided document
                "invoice", "payment summary", "total amount", "booking fees", "ride charge",
                "cancellation policy", "cancellation fees", "cancellation notice", "cancellation confirmation",
            ]

        for document in documents:
            try:
                logger.info(f"Scanning document: {document.title}")

                file_text = document.content

                file_text_lower = file_text.lower()

                for keyword in alert_keywords:
                    if keyword in file_text_lower:
                        idx = file_text_lower.find(keyword)
                        snippet = file_text[max(0, idx-100): idx+100]

                        # Check if already created (optional safety)
                        if not DocumentAlert.objects.filter(document=document, keyword=keyword).exists():
                            DocumentAlert.objects.create(
                                document=document,
                                keyword=keyword,
                                snippet=snippet
                            )
                            logger.info(f"[+] Alert created: {keyword}")

                total_scanned += 1

            except Exception as e:
                logger.error(f"[!] Error scanning document {document.title}: {e}")

        logger.info(f"[✅] Completed scanning {total_scanned} documents.")