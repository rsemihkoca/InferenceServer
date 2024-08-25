import logging

def setup_logger(level, filename):
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=filename)
    return logging.getLogger(__name__)