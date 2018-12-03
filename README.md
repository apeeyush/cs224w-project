CS224W : Analysis of Networks
========

Git repository for CS224W project.

## Setup

    # Download the data
    wget http://aminer.org/lab-datasets/citation/citation-network1.zip
    unzip citation-network1.zip
    # Install relevant python packages
    pip install --user -r requirements.txt
    cd GraphSAGE/
    pip install --user -r requirements.txt

## Running

    # Run parser to generate the data
    python parser.py
    # Run GraphSAGE unsupervised
    cd GraphSAGE
    bash example_unsupervised.sh
    # Run evaluation script
    cd ..
    python eval.py

