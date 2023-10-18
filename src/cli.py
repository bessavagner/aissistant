"""
    This modules provide a class to manage the CLI
    The package's entry point is aissistant.

    To understand more how to implement this module
    check:
     - 
"""
import logging

import click

from genaikit.settings import HEADER
from app import MyApp

logger_client = logging.getLogger('client')


class CLI:
    def __init__(self,):
        self.app = None

    def start(self, arg=None):
        click.echo(HEADER)
        self.app = MyApp(arg)
        # self.app.run()

@click.command()
@click.option('--arg',
              default=None,
              required=False,
              help='Argument to MyApp')
def start_program(arg=None):
    program = CLI()
    program.start(arg)
