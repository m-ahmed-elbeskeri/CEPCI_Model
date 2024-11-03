# src/utils.py
import logging
import yaml

def load_config(path='config/config.yaml'):
    """Load configuration from a YAML file."""
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(level='INFO', filename='logs/forecast.log'):
    """Set up logging to both console and a file."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', filename=filename)
    console = logging.StreamHandler()
    console.setLevel(level)
    logging.getLogger().addHandler(console)
