import logging
import os
import socket
from datetime import datetime

# Unified Logger Utility Class
class Logger:
    def __init__(self, log_dir="logs", log_file="training.log", rank=None, log_level=logging.INFO):
        """
        Initializes a Logger instance with a given directory, file name, rank, and log level.

        Args:
            log_dir (str): Directory where log files are stored.
            log_file (str): Base log file name.
            rank (int): Rank of the current process (for distributed training).
            log_level (int): Logging level (e.g., logging.INFO).
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.rank = rank
        self.log_level = log_level
        self.system_name = socket.gethostname()

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate timestamped log file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rank_suffix = f"_rank_{rank}" if rank is not None else ""
        full_log_path = os.path.join(self.log_dir, f"{timestamp}{rank_suffix}_{self.log_file}")

        # Create logger
        self.logger = logging.getLogger(f"kIvI-2.0_Logger_Rank_{rank if rank is not None else 'main'}")
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Log format (includes rank and system name)
        formatter = logging.Formatter(
            fmt=f"[%(asctime)s] [System: {self.system_name}] [Rank: {self.rank}] - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(full_log_path)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
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

    @staticmethod
    def get_rank_logger(log_dir, log_file, rank):
        """
        Static method to create a logger for distributed training processes.

        Args:
            log_dir (str): Directory where logs will be saved.
            log_file (str): Base name of the log file.
            rank (int): Rank of the current process.

        Returns:
            Logger: Configured logger instance.
        """
        return Logger(log_dir=log_dir, log_file=log_file, rank=rank)

# Example usage for main process and distributed rank-specific logs
if __name__ == "__main__":
    # Logger for main process
    main_logger = Logger(log_dir="logs", log_file="training_main.log")
    main_logger.info("Main process logger initialized.")
    main_logger.debug("This is a debug message.")
    main_logger.warning("This is a warning message.")
    main_logger.error("This is an error message.")
    main_logger.critical("This is a critical message.")

    # Logger for rank-specific processes (example for rank 0)
    rank_logger = Logger.get_rank_logger(log_dir="logs", log_file="training.log", rank=0)
    rank_logger.info("Rank 0 logger initialized.")
    rank_logger.debug("Debug message from Rank 0.")
