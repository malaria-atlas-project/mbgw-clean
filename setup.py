from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('mbgw',parent_package=None,top_path=None)

config.packages = ["mbgw"]

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))