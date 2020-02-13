from datetime import datetime
import logging
from pathlib import Path


class Logger:
    def __init__(self, model_name, log_dir: Path, level=logging.INFO, echo=False):
        self.model_name = model_name
        (log_dir / "summaries").mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = 'log_{}.txt'.format(date_str)
        formatter = logging.Formatter(
            '[%(levelname)s][%(asctime)s] %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S'
        )
        self.logger = logging.getLogger("bot")
        # Remove all existing handlers
        self.logger.handlers = []
        # Initialize handlers
        fh = logging.FileHandler(log_dir / log_file)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        if echo:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        self.logger.setLevel(level)
        self.logger.propagate = False

    def info(self, msg, *args):
        self.logger.info(msg, *args)

    def warning(self, msg, *args):
        self.logger.warning(msg, *args)

    def debug(self, msg, *args):
        self.logger.debug(msg, *args)

    def error(self, msg, *args):
        self.logger.error(msg, *args)
