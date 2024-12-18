import os
import logging
from datetime import datetime

class Logger:
    def __init__(self, log_file, rank=0):
        """
        Custom logger to handle logs with timestamp and rank-specific information.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        full_log_path = os.path.join(
            os.path.splitext(log_file)[0] + f"_{timestamp}_rank_{rank}.log"
        )

        # Ensure the directory for logs exists
        os.makedirs(os.path.dirname(full_log_path), exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger(f"Logger_{rank}")
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(full_log_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
