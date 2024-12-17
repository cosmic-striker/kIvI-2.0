import logging
import os
import socket
from datetime import datetime

# Logger Utility Class
class Logger:
    def __init__(self, log_dir="logs", log_file="training.log", rank=0, log_level=logging.INFO):
        """
        Initializes a Logger instance with directory, file name, rank, and log level.
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_level = log_level
        self.rank = rank
        self.system_name = socket.gethostname()

        # Ensure log directory exists
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.log_dir, f"{timestamp}_rank_{rank}_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Complete log file path
        full_log_path = os.path.join(self.log_dir, self.log_file)

        # Logger Setup
        self.logger = logging.getLogger(f"rank_{rank}")
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File Handler
        file_handler = logging.FileHandler(full_log_path)
        file_handler.setLevel(self.log_level)
        formatter = logging.Formatter(
            f"[%(asctime)s] [System: {self.system_name}] [Rank: {rank}] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)

        # Add Handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        """Log an INFO level message."""
        self.logger.info(message)

    def error(self, message):
        """Log an ERROR level message."""
        self.logger.error(message)

    def warning(self, message):
        """Log a WARNING level message."""
        self.logger.warning(message)

    def debug(self, message):
        """Log a DEBUG level message."""
        self.logger.debug(message)

    def critical(self, message):
        """Log a CRITICAL level message."""
        self.logger.critical(message)
