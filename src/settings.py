"""
    This module provides the default settings of the
    package test
"""

from importlib import metadata

package = metadata.metadata('aissistant')

name = package['name']
version = package['version']
author = package['author']
author_email = package['author-email']
summary = package['summary']

TITLE = name
DELIMITER = len(TITLE)*"="
HEADER = f"""
{DELIMITER}
{TITLE}
Version: {version}
Description: {summary }
Authors: {author}
{DELIMITER}
"""

CONFIG_LOG = {
    "version": 1,
    "formatters": {
        "client": {"format": "%(levelname)s: %(message)s"},
        "standard": {
            "format": "%(levelname)s (at %(module)s.%(funcName)s in line %(lineno)d): %(message)s"
        },
        "debug": {
            "format": (
                "%(levelname)s (at %(module)s.%(funcName)s in line %(lineno)d):"
                "\n\t|──file: %(pathname)s"
                "\n\t|──process: %(process)d | name: %(processName)s"
                "\n\t|──thread: %(thread)d | name: %(threadName)s"
                "\n\t└──message: %(message)s\n"
            ),
        },
        "file": {
            "format": (
                "%(levelname)s %(asctime)s (at %(module)s.%(funcName)s in line %(lineno)d): %(message)s"
                "\n\t|──file: %(pathname)s\n\t"
                "\n\t|──process: %(process)d | name: %(processName)s"
                "\n\t|──thread: %(thread)d | name: %(threadName)s\n"
                "\n\t└──message: %(message)s\n"
            ),
            "datefmt": "%y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "client": {
            "class": "logging.StreamHandler",
            "formatter": "client",
            "level": "INFO",
        },
        "standard": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "DEBUG",
        },
        "debug": {
            "class": "logging.StreamHandler",
            "formatter": "debug",
            "level": "DEBUG",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "file",
            "level": "DEBUG",
            "filename": "report.log",
            "encoding": "utf8",
        },
    },
    "root": {"handlers": ["standard"], "level": "DEBUG"},
    "loggers": {
        "client": {
            "handlers": ["client"],
            "level": "DEBUG",
            "propagate": False,
            "disable_existing_loggers": False
        },
        "standard": {
            "handlers": ["standard"],
            "level": "DEBUG",
            "propagate": False,
            "disable_existing_loggers": False
        },
        "debugger": {
            "handlers": ["debug"],
            "level": "DEBUG",
            "propagate": False,
            "disable_existing_loggers": False
        },
        "report": {
            "handlers": ["file"],
            "level": "DEBUG",
            "propagate": False,
            "disable_existing_loggers": False
        },
        "report_debugger": {
            "handlers": ["file", "debug"],
            "level": "DEBUG",
            "propagate": False,
            "disable_existing_loggers": False
        },
    },
}
