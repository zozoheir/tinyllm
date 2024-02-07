#!/bin/bash
export TINYLLM_CONFIG_PATH = '/Users/othmanezoheir/PycharmProjects/openagents/tinyllm/tinyllm/config/config.json'
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/personal/smartpy"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-backend"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-env"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-jobs"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-models"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-pyenv"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/rumorz-io/rumorz-llms"
PYTHONPATH="$PYTHONPATH:/Users/othmanezoheir/PycharmProjects/personal/langdt"

export PYTHONPATH


# Set up a virtual environment path for ease of use
VENV_PATH="/Users/othmanezoheir/venv/rumorz-jobs-2/bin/python"
cd /Users/othmanezoheir/PycharmProjects/openagents/tinyllm
# Run unittest discovery in the tinyllm/tests directory
if $VENV_PATH -m unittest discover tinyllm; then
    # If tests pass, proceed to add, commit, and push changes to git
    git add .
    echo "Enter commit message:"
    read commit_message
    git commit -m "[$commit_message]"
    if [ $? -eq 0 ]; then
        git push
        echo "Changes pushed successfully!"
    else
        echo "Commit failed, changes not pushed."
    fi
else
    # If tests fail, halt the process
    echo "Tests failed, changes not added or committed."
fi
