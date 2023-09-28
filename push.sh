#!/bin/bash

# We have to run files 1 by 1 due to langfuse hanging program issue

# Running unittests for functions and chains folder
if /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest discover -p 'test_*.py' tinyllm/tests/functions tinyllm/tests/chains; then
  for file in tinyllm/tests/llms/*.py; do
      # Check if it's a file and not a directory
      if [[ -f "$file" ]]; then
          # Run the tests for that file
          if ! /Users/othmanezoheir/venv/rumorz-jobs-2/bin/python -m unittest "$file"; then
              echo "Test $file failed, stopping."
              exit 1
          fi
      fi
  done

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
