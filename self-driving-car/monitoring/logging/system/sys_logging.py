# System logging code
import logging

def setup_system_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('system')
    return logger
