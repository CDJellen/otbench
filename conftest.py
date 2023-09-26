from os import path

import pandas as pd
import pytest

from otb.tasks import TaskApi


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption('--run-slow', action='store_true', default=False, help='execute long-running tests.')
    parser.addoption('--run-private', action='store_true', default=False, help='run tests for private methods.')


def pytest_configure(config):
    """Add custom markers to pytest."""
    config.addinivalue_line('markers', 'slow: flag tests as slow to run')
    config.addinivalue_line('markers', 'private: flag tests for private methods')


def pytest_collection_modifyitems(config, items):
    """Skip slow tests and tests for private methods by default."""
    flags_to_skip = set(['slow', 'private'])
    if config.getoption('--run-slow'):
        # --run-slow given in cli: do not skip long running tests
        flags_to_skip.remove('slow')
    if config.getoption('--run-private'):
        # --run-private given in cli: do not skip tests of private methods
        flags_to_skip.remove('private')
    if not flags_to_skip:
        return

    skip_slow = pytest.mark.skip(reason='use `--run-slow` option to run')
    skip_private = pytest.mark.skip(reason='use `--run-private` option to run')
    
    for item in items:
        if 'slow' in item.keywords and 'slow' in flags_to_skip:
            item.add_marker(skip_slow)
        if 'private' in item.keywords and 'private' in flags_to_skip:
            item.add_marker(skip_private)


@pytest.fixture(scope="session")
def task_api():
    """Return a function that yields a pointer to the TaskApi."""
    api = TaskApi()
    yield api
