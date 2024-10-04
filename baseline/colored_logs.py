#!/usr/bin/env python

# From https://stackoverflow.com/a/46482050

from copy import copy
import logging
from logging import Formatter

MAPPING = {
  'DEBUG'   : 37, # white
  'INFO'    : 32, # cyan
  'WARNING' : 33, # yellow
  'ERROR'   : 31, # red
  'CRITICAL': 41, # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'

class ColoredFormatter(Formatter):
  def __init__(self, pattern):
    Formatter.__init__(self, pattern, datefmt='%Y-%m-%d %H:%M:%S')

  def format(self, record):
    colored_record = copy(record)
    levelname = colored_record.levelname
    seq = MAPPING.get(levelname, 37) # default white
    colored_levelname = ('{0}{1}m{2}{3}').format(PREFIX, seq, levelname, SUFFIX)
    colored_record.levelname = colored_levelname
    return Formatter.format(self, colored_record)
  
def setup_logger():
  global logger

  logger = logging.getLogger("main")
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  cf = ColoredFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
  ch.setFormatter(cf)
  logger.addHandler(ch)
  logger.setLevel(logging.INFO)

  return logger