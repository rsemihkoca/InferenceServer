import logging

def setup_logger(level, filename):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create handlers
    file_handler = logging.FileHandler(filename)
    console_handler = logging.StreamHandler()

    # Set levels for handlers (optional, could be different from the logger's level)
    file_handler.setLevel(level)
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
