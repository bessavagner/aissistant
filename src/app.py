"""
    Package aissistant

    This can be an sample app using the package
    or the app itself to feed the __main__ and gui
    modules.
"""

import logging
from logging.config import dictConfig

from aissistant import CONFIG_LOG

dictConfig(CONFIG_LOG)

logger = logging.getLogger('client')
report = logging.getLogger('report')

class MyApp:
    def __init__(self, *args, **kwargs):
        pass
