import logging
import os

def setup_logger(level, filename):
    # Check if the file exists, and if so, delete it
    if os.path.exists(filename):
        os.remove(filename)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    file_handler = logging.FileHandler(filename)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
