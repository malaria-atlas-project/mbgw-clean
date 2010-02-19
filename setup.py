from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('generic_afgh',parent_package=None,top_path=None)
config.add_extension(name='cf_helper',sources=['generic_afgh/cf_helper.f'])

config.packages = ["generic_afgh"]

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))