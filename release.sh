#!/bin/bash

# Run the commit.sh script
./commit.sh

# Check if commit.sh was successful
if [ $? -eq 0 ]; then
    # Only proceed if commit.sh was successful

    source "/Users/othmanezoheir/venv/rumorz-jobs-2/bin/activate"
    # Run unittest discovery in the tinyllm/tests directory
    bump2version patch  # for a patch version increment
    python setup.py sdist bdist_wheel
    twine upload dist/*
    echo "New version released!"
else
    # If commit.sh failed, do not proceed
    echo "Commit failed, release process aborted."
fi
