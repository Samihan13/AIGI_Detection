# utils/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')  # Ensure there is a directory to store log files

    logger = logging.getLogger('AIImageDetection')
    logger.setLevel(logging.DEBUG)  # Adjust as per the verbosity requirement

    # Console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)



    # File handler which logs even debug messages
    fh = RotatingFileHandler('logs/ai_image_detection.log', maxBytes=1024*1024*5, backupCount=5)
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter_console = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    formatter_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter_console)
    fh.setFormatter(formatter_file)

    # Add the handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
