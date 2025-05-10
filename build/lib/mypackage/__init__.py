"""
mypackage
=========

Available subpackages
---------------------
utils
    utility functions
"""
# help with 2to3 support.
from __future__ import absolute_import, division, print_function

from . import version
from .version import __version__

from . import core
from .core import *

from . import utils
