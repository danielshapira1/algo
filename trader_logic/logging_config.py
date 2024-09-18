import logging
import os
from trader_logic.trader_utils import get_project_root

def setup_logging():
    log_dir = os.path.join(get_project_root(), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Remove all handlers from the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, "trader.log")),
                            logging.StreamHandler()
                        ])

    # Create a separate handler for debug logs
    debug_handler = logging.FileHandler(os.path.join(log_dir, "trader_debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)

    # Add the debug handler to the root logger
    logging.getLogger().addHandler(debug_handler)