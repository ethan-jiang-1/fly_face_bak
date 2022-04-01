#!/bin/bash
find . -name "*.blend1" -exec rm -rf {} \;
find . -name "*.pyc" -exec rm -rf {} \;
find . -name "__pycache__" -exec rm -rf {} \;