import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "AWESim_SOSS",
    version = "0.0.5",
    author = "Joe Filippazzo and Jonathan Fraine",
    author_email = "jfilippazzo@stsci.edu",
    description = ("Analyzing Webb Exoplanet Simulations with SOSS"),
    license = "MIT",
    keywords = "",#"NIRISS SOSS Exoplanet Atmospheres",
    url = "https://github.com/ExoCTK/AWESim_SOSS",
    packages=['AWESim_SOSS'],#, 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT",
    ],
)