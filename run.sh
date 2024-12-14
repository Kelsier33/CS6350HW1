#!/bin/bash

# Navigate to the Logistic Regression directory and run lr_svm_comparison.py
echo "Running Logistic Regression and SVM comparison script..."
cd Logistic\ Regression
python3 lr_svm_comparison.py
cd ..

# Navigate to the Neural Networks directory and run ff_neural_network.py
echo "Running Feed-Forward Neural Network script..."
cd Neural\ Networks
python3 ff_neural_network.py
cd ..

echo "All scripts executed successfully!"
