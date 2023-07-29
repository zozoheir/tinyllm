#!/bin/bash
cd /Users/othmanezoheir/PycharmProjects/openagents/tiny-llm

if /Users/othmanezoheir/venv/tiny-llm/bin/Python -m unittest discover; then
  git add .
  echo "Enter commit message:"
  read commit_message
  git commit -m "$commit_message"
  if [ $? -eq 0 ]; then
    git push
    echo "Changes pushed successfully!"
  else
    echo "Commit failed, changes not pushed."
  fi
else
  echo "Tests failed, changes not added or committed."
fi
