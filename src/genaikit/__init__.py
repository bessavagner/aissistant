"""
    Package aissistant

    <Write the package's description here>
"""

import logging
from logging import NullHandler
from logging.config import dictConfig

from settings import CONFIG_LOG

from . import core  # The core module is the packages's API
from . import base
from . import data
from . import utils
from . import constants

dictConfig(CONFIG_LOG)

# Set default logging handler to avoid \"No handler found\" warnings.
logging.getLogger(__name__).addHandler(NullHandler())

__all__ = [
    'constants',
    'core',
    'base',
    'data',
    'utils',
]
