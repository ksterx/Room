# MIT License
#
# Copyright (c) 2021 Toni-SM
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os

__all__ = ["__version__", "logger"]

# read library version from file
path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(path, "r") as file:
    __version__ = file.read().strip()


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {
        logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
        logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
        logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
        logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
        logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m",
    }

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)


_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("room")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
