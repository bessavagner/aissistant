"""
    Package aissistant

    <Write the package's description here>
"""

import logging
from logging import NullHandler
from logging.config import dictConfig

from genaikit.settings import CONFIG_LOG

from . import core  # The core module is the packages's API
from . import AsyncCore  # The core module is the packages's API
from . import base
from . import AsyncBase
from . import utils
from . import constants
from . import nlp

from .core import (
    Chatter,
    MiniChatter,
    AdvancedChatter,
    Context,
    QuestionContext
)
from .AsyncCore import (
    AsyncChatter,
    AsyncMiniChatter,
    AsyncAdvancedChatter,
    AsyncContext,
    AsyncQuestionContext
)

dictConfig(CONFIG_LOG)

# Set default logging handler to avoid \"No handler found\" warnings.
logging.getLogger(__name__).addHandler(NullHandler())

__all__ = [
    'constants',
    'core',
    'AsyncCore',
    'base',
    'AsyncBase',
    'utils',
    'nlp'
]
