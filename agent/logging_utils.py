from torch import distributed as dist
import logging
import os
import functools
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from datetime import datetime, timedelta

logger_initialized = {}


class FixedTimeFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_time = datetime.now()
        self.fixed_time_str = self.fixed_time.strftime("%Y-%m-%d %H:%M:%S")

    def formatTime(self, record, datefmt=None):
        record_time = datetime.fromtimestamp(record.created)
        elapsed_time = record_time - self.fixed_time
        elapsed_days = elapsed_time.days
        elapsed_str = str(elapsed_time)

        return f"{self.fixed_time_str}, {elapsed_days}:{elapsed_str}"


class FixedTimeFormatterText:
    def __init__(self):
        self.fixed_time = datetime.now()

    def format_time(self):
        current_time = datetime.now()
        elapsed_time = current_time - self.fixed_time
        elapsed_seconds = elapsed_time.total_seconds()
        hours, remainder = divmod(int(elapsed_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        microseconds = elapsed_time.microseconds

        # Format the elapsed time
        formatted_time = f"{hours}:{minutes:02}:{seconds:02}.{microseconds:06}"
        return f"{self.fixed_time.strftime('%Y-%m-%d %H:%M:%S')} {formatted_time} "


class RichLogger:

    @functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
    def __init__(self, name: str, level: int = logging.INFO, log_file: str = "app.log"):
        self.name = name
        custom_theme = Theme(
            {
                "info": "white",
                "warning": "magenta",
                "debug": "cyan",
                "danger": "bold red",
                "error": "bold red",
            }
        )
        # self.logger = Logger(name, level, log_file)
        self.console = Console(theme=custom_theme, record=True)
        self.log_file = log_file

    def log(self, level, message):
        self.console.log(
            self.name
            + " - "
            + self.formatter.format_time()
            + " ".join(str(m) for m in message),
            style="info",
        )

    def debug(self, message: str):
        self.formatter = FixedTimeFormatterText()
        self.console.log(self.formatter.format_time() + message, style="debug")

    def info(self, *message: str):
        self.formatter = FixedTimeFormatterText()
        self.console.log(
            self.name
            + " - "
            + self.formatter.format_time()
            + " ".join(str(m) for m in message),
            style="info",
        )
        text = self.console.export_text()
        with open(self.log_file, "a+", encoding="utf-8") as f:
            f.write(text)

    def warning(self, message: str):
        self.formatter = FixedTimeFormatterText()
        self.console.log(self.formatter.format_time() + message, style="warning")

    def error(self, message: str):
        self.formatter = FixedTimeFormatterText()
        self.console.log(self.formatter.format_time() + message, style="error")

    def critical(self, message: str):
        self.formatter = FixedTimeFormatterText()
        self.console.log(self.formatter.format_time() + message, style="danger")


class Logger:

    @functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
    def __init__(self, name: str, level: int = logging.INFO, log_file: str = "app.log"):

        custom_theme = Theme(
            {"info": "dim green", "warning": "magenta", "danger": "bold red"}
        )
        self.console = Console(theme=custom_theme)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        # self.logger.propagate = False


        handler = RichHandler(console=self.console, markup=True)
        handler.setLevel(level)


        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)


        formatter = FixedTimeFormatter(
            "%(message)s"
        )  # "%(name)s - %(asctime)s - %(message)s")

        handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.handlers.clear()

        # self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)

    def log(self, level, msg):
        self.logger.log(level, msg)

    def debug(self, *message):
        # Convert all message items to strings
        self.logger.debug(" ".join(str(m) for m in message))

    def info(self, *message):
        # Convert all message items to strings
        self.logger.info(" ".join(str(m) for m in message))
        # self.logger.info([m for m in message])

    def warning(self, *message):
        # Convert all message items to strings
        self.logger.warning(" ".join(str(m) for m in message))

    def error(self, *message):
        # Convert all message items to strings
        self.logger.error(" ".join(str(m) for m in message))

    def critical(self, *message):
        # Convert all message items to strings
        self.logger.critical(" ".join(str(m) for m in message))


def get_logger(name=None, work_dir=None, cfg=None, log_level=logging.INFO):
    if name in logger_initialized:
        return logger_initialized[name]
    else:
        raise ValueError(f"Logger {name} not initialized")


def create_logger(experimental_desc, work_dir=None, cfg=None, log_level=logging.INFO):

    if work_dir is not None:
        # time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # work_dir = work_dir + "/" + time_str
        final_log_file = os.path.join(work_dir, f"{experimental_desc}.log")
        os.makedirs(work_dir, exist_ok=True)
        logger = Logger(experimental_desc, log_level, log_file=final_log_file)
        logger_initialized[experimental_desc] = logger

        return logger


def print_log(msg, logger=None, level=logging.INFO, clear_logger=False):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif hasattr(logger, "log"):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )
    if clear_logger:
        logger_initialized = {}


