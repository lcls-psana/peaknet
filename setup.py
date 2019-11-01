#coding: utf8

"""
Setup script for peaknet
"""

from glob import glob
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='peaknet',
      version='0.1',
      author='Ponan Li',
      author_email="liponan@slac.stanford.edu",
      description="neural network for peak detection",
      packages=["peaknet"],
      package_dir={"peaknet": "peaknet"},
      scripts=[s for s in glob('scripts/*') if not s.endswith('__.py')]
)
