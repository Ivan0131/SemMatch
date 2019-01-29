from distutils.core import setup
from setuptools import find_packages

# This is a list of files to install, and where
# (relative to the 'root' dir, where setup.py is)
# You could be more specific.

setup(name="semmatch",
      version="0.1",
      #description="****",
      #author="****",
      #author_email="****",
      #url="****",
      packages=find_packages(where='.', exclude=(), include=('*',)),
      #package_data={'package': files},
      #long_description="""*****"""
      )
