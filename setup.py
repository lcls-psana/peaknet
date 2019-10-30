try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='peaknet',
      version='0.0.1',
      author="Ponan Li",
      author_email="liponan@slac.stanford.edu",
      description='ML peakfinding for LCLS',
      url='https://github.com/lcls-psana/peaknet',
      packages=["peaknet"],
      package_dir={"peaknet": "."})
