import logging

logger = logging.getLogger('aws_diffusion_model')
logger.setLevel(logging.INFO)

# Prevent duplicate handlers during tests or reruns
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
