#!/bin/bash

# Running unittests for functions and chains folder
if /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest discover -p 'test_*.py' tinyllm/tests/functions tinyllm/tests/chains; then

  if ! /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest tinyllm/tests/test_vector_store.py; then
    echo "Test test_vector_store.py failed, stopping."
    exit 1
  fi

  # Running specific tests 1 by 1 because of Langfuse hanging issue
  if ! /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest tinyllm/tests/llms/test_memory.py; then
    echo "Test test_memory.py failed, stopping."
    exit 1
  fi

  if ! /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest tinyllm/tests/llms/test_openai_agent.py; then
    echo "Test test_openai_agent.py failed, stopping."
    exit 1
  fi

  if ! /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest tinyllm/tests/llms/test_openai_chat.py; then
    echo "Test test_openai_chat.py failed, stopping."
    exit 1
  fi

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
  echo "Tests failed, changes not added or committed."
fi
