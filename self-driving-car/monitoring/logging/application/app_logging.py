# Application logging code
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('self_driving_car')
    return logger
