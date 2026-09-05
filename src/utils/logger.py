import logging

BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
RED = "\033[91m"


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: YELLOW,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        prefix = (
            f"{self.formatTime(record, self.datefmt)} "
            f"[{record.levelname}]"
        )
        message = record.getMessage()

        return f"{color}{prefix}{RESET} {message}"


class CallMeLogger:
    """Custom logger for the CallMeMaybe application.
    Attributes:
        logger: The underlying logger instance.
    Methods:
        set_level: Set the logging level.
        debug: Log a debug message.
        info: Log an info message.
        warning: Log a warning message.
        error: Log an error message.
        critical: Log a critical message.
    """

    def __init__(self, level: str = "INFO"):
        self.logger = logging.getLogger("CallMeMaybe")
        self.set_level(level)
        ch = logging.StreamHandler()

        formatter = ColorFormatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def set_level(self, level: str):
        """Set the logging level.

        Args:
            level: Logging level as a string (e.g., "DEBUG", "INFO").
        """
        if level.upper() == "DEBUG":
            numeric_level = logging.DEBUG
        else:
            numeric_level = logging.INFO
        self.logger.setLevel(numeric_level)

    def debug(self, msg: object) -> None:
        self.logger.debug(msg)

    def info(self, msg: object) -> None:
        self.logger.info(msg)

    def warning(self, msg: object) -> None:
        self.logger.warning(msg)

    def error(self, msg: object) -> None:
        self.logger.error(msg)

    def critical(self, msg: object):
        self.logger.critical(msg)
