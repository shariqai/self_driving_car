# Audit logging code
import logging

def setup_audit_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('audit')
    return logger
