#!/bin/bash

# Check if the script is run with a parameter
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <WANDB_API_KEY>"
  exit 1
fi

# Get the API key from the user input
WANDB_API_KEY=$1

# Define the utils folder and file path
UTILS_FOLDER="utils"
FILE_PATH="$UTILS_FOLDER/wandb_private.py"

# Write the WANDB_API_KEY line to the file
echo "WANDB_API_KEY = '$WANDB_API_KEY'" > "$FILE_PATH"

# Inform the user
if [ $? -eq 0 ]; then
  echo "File created successfully: $FILE_PATH"
else
  echo "Failed to create the file: $FILE_PATH"
  exit 1
fi