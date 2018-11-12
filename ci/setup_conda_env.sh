#!/bin/bash

echo "Creating a Python $PYTHON_VERSION environment"
conda create -n awesimsoss python=$PYTHON_VERSION || exit 1
source activate awesimsoss

echo "Installing packages..."
conda install flake8 beautifulsoup4 lxml numpy astropy
pip install exoctk pytest pytest-cov coveralls