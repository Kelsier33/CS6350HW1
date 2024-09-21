#!/bin/bash

# Load Python if necessary (if you're not using a virtual environment with python installed)
# Uncomment the following if required:
# module load python

# Path to the Python script
PYTHON_SCRIPT="id3_algorithm.py"

# Ensure the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "Python script '$PYTHON_SCRIPT' not found!"
  exit 1
fi

# Set paths for training and test datasets
CAR_TRAIN="train.csv"          # Car dataset training file
CAR_TEST="test.csv"            # Car dataset test file
BANK_TRAIN="train_.csv"        # Bank dataset training file
BANK_TEST="test_.csv"          # Bank dataset test file

# Ensure datasets exist
if [[ ! -f "$CAR_TRAIN" || ! -f "$CAR_TEST" || ! -f "$BANK_TRAIN" || ! -f "$BANK_TEST" ]]; then
  echo "One or more dataset files not found!"
  exit 1
fi

# Set the maximum depth values for testing (1 through 6)
MAX_DEPTH_VALUES=(1 2 3 4 5 6)

# Run the experiment on the Car dataset
echo "Running experiment on the Car dataset (max depth 1-6, all metrics)..."
python3 "$PYTHON_SCRIPT" --train "$CAR_TRAIN" --test "$CAR_TEST" --max-depth 6 --metrics "entropy" "gini" "majority_error"

# Run the experiment on the Bank dataset with "unknown" treated as missing (max depth 1-6, all metrics)
echo "Running experiment on the Bank dataset with 'unknown' as missing (max depth 1-6, all metrics)..."
python3 "$PYTHON_SCRIPT" --train "$BANK_TRAIN" --test "$BANK_TEST" --max-depth 6 --metrics "entropy" "gini" "majority_error" --handle-unknown-as-missing

# Run the experiment on the Bank dataset with "unknown" treated as a valid value (max depth 1-6, all metrics)
echo "Running experiment on the Bank dataset with 'unknown' as valid (max depth 1-6, all metrics)..."
python3 "$PYTHON_SCRIPT" --train "$BANK_TRAIN" --test "$BANK_TEST" --max-depth 6 --metrics "entropy" "gini" "majority_error" --handle-unknown-as-valid

echo "All tests completed!"
