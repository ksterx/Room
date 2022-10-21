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

import os

__all__ = ["__version__", "log"]

# read library version from file
_path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(_path, "r") as file:
    __version__ = file.read().strip()


class Logger:
    _grey = "\x1b[37;40m"
    _blue = "\x1b[34;20m"
    _yellow = "\x1b[33;20m"
    _green = "\x1b[32;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"

    def debug(self, *msg):
        msg = " ".join(msg)
        print(f"{self._blue}[DEBUG] {msg}{self._reset}")

    def info(self, *msg):
        msg = " ".join(msg)
        print(f"{self._grey}[INFO] {msg}{self._reset}")

    def warning(self, *msg):
        msg = " ".join(msg)
        print(f"{self._yellow}[WARNING] {msg}{self._reset}")

    def error(self, *msg):
        msg = " ".join(msg)
        print(f"{self._red}[ERROR] {msg}{self._reset}")

    def critical(self, *msg):
        msg = " ".join(msg)
        print(f"{self._bold_red}[CRITICAL] {msg}{self._reset}")


log = Logger()
