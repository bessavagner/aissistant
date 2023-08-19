"""
    Package "aissistant"

    This module can be a sample use of the package
    or the app structure to feed the cli module.
"""

import logging
from logging import NullHandler

from cli import start_program

# Set default logging handler to avoid \"No handler found\" warnings.
logging.getLogger(__name__).addHandler(NullHandler())

def main():
    start_program()

if __name__ == '__main__':
    main()
