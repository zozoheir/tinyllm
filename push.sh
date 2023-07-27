#!/bin/bash
cd /Users/othmanezoheir/PycharmProjects/openagents/tiny-llm

# Run the tests and check if they are successful
if /Users/othmanezoheir/venv/tiny-llm/bin/Python -m unittest discover; then
  # If the tests pass, start the git process
  git add .

  # Ask for a commit message
  echo "Enter commit message:"
  read commit_message

  # Commit the changes
  git commit -m "$commit_message"

  # Check if the commit was successful
  if [ $? -eq 0 ]; then
    # Push the changes
    git push
    echo "Changes pushed successfully!"
  else
    echo "Commit failed, changes not pushed."
  fi
else
  echo "Tests failed, changes not added or committed."
fi
