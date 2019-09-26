import logging

log_format = """{"time": "%(asctime)s","level": "%(levelname)s",
"log_name": "%(name)s", "function": "%(funcName)s","message": "%(message)s"}"""\
    .strip().replace('\n','')


class Logger(object):
    def __init__(self, name, level="INFO"):
        self.logger = logging.getLogger(name)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(level)

    def get_logger(self) -> logging.Logger:
        return self.logger
