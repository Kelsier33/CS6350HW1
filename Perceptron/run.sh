#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit
fi

# Define the paths to the perceptron files
FILES=("perceptron_average.py" "perceptron_standard.py" "perceptron_voted.py")

# Run each perceptron file
for file in "${FILES[@]}"
do
    if [ -f "$file" ]; then
        echo "Running $file..."
        python3 "$file"
        echo "$file completed."
        echo "-----------------------------"
    else
        echo "$file not found."
    fi
done

echo "All perceptron scripts have been executed."
