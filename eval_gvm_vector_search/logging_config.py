"""Logging configuration using Loguru"""

import sys
import logging
import inspect
from loguru import logger

# Whitelist: only allow logs from these modules
ALLOWED_LOGGERS = [
    'eval_gvm_vector_search',
    '__main__',
]


def setup_logging(level: str = "INFO"):
    """
    Configure Loguru logger with whitelist approach - only show application logs.
    
    Args:
        level: Log level (default: INFO)
    """

    # Intercept standard logging (official loguru approach)
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Whitelist: only allow logs from application modules
            if not any(record.name.startswith(allowed) for allowed in ALLOWED_LOGGERS):
                return
            
            # Get corresponding Loguru level
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller (official loguru method)
            frame, depth = inspect.currentframe(), 0
            while frame:
                filename = frame.f_code.co_filename
                is_logging = filename == logging.__file__
                is_frozen = "importlib" in filename and "_bootstrap" in filename
                if depth > 0 and not (is_logging or is_frozen):
                    break
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Apply interception
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)
    
    # Set root logger to CRITICAL to suppress everything else
    logging.getLogger().setLevel(logging.CRITICAL)
