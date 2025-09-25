"""
Logging utilities for post-editing pipeline.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger('post_editing')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_processing_stats(logger: logging.Logger, stats: dict):
    """Log processing statistics."""
    logger.info("="*50)
    logger.info("PROCESSING STATISTICS")
    logger.info("="*50)
    
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"{key.upper()}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    logger.info("="*50)


def log_model_info(logger: logging.Logger, model_type: str, model_name: str, config: dict):
    """Log model initialization information."""
    logger.info(f"Initializing {model_type} model: {model_name}")
    logger.info("Model configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
