import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def trace_span(name: str, **metadata):
    start = time.time()
    logger.info(f"TRACE START {name} | {metadata}")
    try:
        yield
        duration = time.time() - start
        logger.info(f"TRACE END {name} | duration={duration:.2f}s | {metadata}")
    except Exception as e:
        duration = time.time() - start
        logger.exception(f"TRACE ERROR {name} | duration={duration:.2f}s | {metadata} | error={e}")
        raise
