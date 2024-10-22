# CS6350HW1
This is a machine learning library developed by Hyrum Bailey for CS5350/6350 at the University of Utah.

## How to Use the Code to Learn Decision Trees

### Running the Decision Tree Code Using `run.sh`
- Run the script by using the command:
  ```bash
  ./run.sh

This script will run the id3_algorithm.py script with different configurations automatically.

You can run the decision tree learning directly using id3_algorithm.py. For example:
Parameters:
--train_path: Path to the training dataset (e.g., train.csv).
--test_path: Path to the test dataset (e.g., test.csv).
--metric: Splitting criterion, can be entropy, gini, or majority_error.
--max_depth: Maximum depth of the decision tree (e.g., 3).