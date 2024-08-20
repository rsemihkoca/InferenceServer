import logging
from utils.config import load_config

def setup_logger():
    config = load_config()
    logging.basicConfig(
        level=config['logging']['level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config['logging']['file']
    )
    return logging.getLogger(__name__)