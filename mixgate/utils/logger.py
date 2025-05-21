from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, log_path):
        self.log = open(log_path, 'w')

    def write(self, txt):
        self.log.write(txt)
        self.log.flush()

    def close(self):
        self.log.close()

