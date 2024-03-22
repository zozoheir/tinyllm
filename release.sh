#!/bin/bash

# Run the commit.sh script
#./commit.sh

if [ $? -eq 0 ]; then
    echo "Commit successful, proceeding to activate virtual environment."
else
    echo "Commit failed, release process aborted."
    exit 1
fi

# Activate virtual environment
source "/Users/othmanezoheir/venv/rumorz-jobs-2/bin/activate"

if [ $? -eq 0 ]; then
    echo "Virtual environment activated successfully."
else
    echo "Failed to activate virtual environment, release process aborted."
    exit 1
fi

# Increment version number
bump2version patch

if [ $? -eq 0 ]; then
    echo "Version number incremented successfully."
else
    echo "Failed to increment version number, release process aborted."
    exit 1
fi

# Clearing the dist/ directory
echo "Cleaning up the dist/ directory..."
rm -rf dist/*


# Build the package
python setup.py sdist bdist_wheel

if [ $? -eq 0 ]; then
    echo "Package built successfully."
elseO
    echo "Failed to build package, release process aborted."
    exit 1
fi

# Upload to PyPI
twine upload dist/*

if [ $? -eq 0 ]; then
    echo "Package uploaded successfully. New version released!"
else
    echo "Failed to upload package, release process aborted."
    exit 1
fi
