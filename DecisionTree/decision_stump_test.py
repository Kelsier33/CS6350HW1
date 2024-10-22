import math
import csv
from collections import Counter
import argparse
import numpy as np  # For handling weights
import pandas as pd

# Load dataset from a CSV file
def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data[1:]  # Skip header row


# Function to calculate the entropy of a dataset
def entropy(subset, weights):
    total = np.sum(weights)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    entropy_value = 0
    for label, count in label_counts.items():
        weight_sum = np.sum([weights[i] for i, row in enumerate(subset) if row[-1] == label])
        prob = weight_sum / total
        entropy_value -= prob * math.log2(prob) if prob > 0 else 0
    
    return entropy_value

# Function to calculate the Gini Index of a dataset
def gini_index(subset, weights):
    total = np.sum(weights)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    gini = 1.0
    for label, count in label_counts.items():
        weight_sum = np.sum([weights[i] for i, row in enumerate(subset) if row[-1] == label])
        prob = weight_sum / total
        gini -= prob ** 2
    
    return gini

# Function to calculate the Majority Error of a dataset
def majority_error(subset, weights):
    total = np.sum(weights)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    majority_label_weight = max([np.sum([weights[i] for i, row in enumerate(subset) if row[-1] == label]) for label in label_counts])
    majority_class_proportion = majority_label_weight / total
    
    return 1 - majority_class_proportion

# Function to calculate information gain based on the selected metric and using weighted examples
def information_gain(dataset, attribute_index, weights, metric='entropy'):
    if metric == 'entropy':
        total_metric = entropy(dataset, weights)
    elif metric == 'gini':
        total_metric = gini_index(dataset, weights)
    elif metric == 'majority_error':
        total_metric = majority_error(dataset, weights)
    
    total_size = np.sum(weights)
    attribute_values = set(row[attribute_index] for row in dataset)
    
    weighted_metric = 0
    for value in attribute_values:
        subset = [row for i, row in enumerate(dataset) if row[attribute_index] == value]
        subset_weights = [weights[i] for i, row in enumerate(dataset) if row[attribute_index] == value]
        weight = np.sum(subset_weights) / total_size
        if metric == 'entropy':
            weighted_metric += weight * entropy(subset, subset_weights)
        elif metric == 'gini':
            weighted_metric += weight * gini_index(subset, subset_weights)
        elif metric == 'majority_error':
            weighted_metric += weight * majority_error(subset, subset_weights)
    
    return total_metric - weighted_metric

# ID3 algorithm modified to learn decision stumps (single split)
def id3_stump(dataset, attributes, weights, metric='entropy'):
    labels = [row[-1] for row in dataset]
    
    # If all labels are the same, return the label
    if len(set(labels)) == 1:
        return labels[0]
    
    # Select the best attribute to split on (max information gain)
    best_attribute_index = max(range(len(attributes)), key=lambda i: information_gain(dataset, i, weights, metric))
    
    # Create a decision stump (one split)
    tree = {attributes[best_attribute_index]: {}}
    
    # Split dataset on the best attribute and create leaf nodes (no further splits)
    attribute_values = set(row[best_attribute_index] for row in dataset)
    for value in attribute_values:
        subset = [row for row in dataset if row[best_attribute_index] == value]
        subset_weights = [weights[i] for i, row in enumerate(dataset) if row[best_attribute_index] == value]
        
        # Create a leaf node for this split
        leaf_label = Counter([row[-1] for row in subset]).most_common(1)[0][0]
        tree[attributes[best_attribute_index]][value] = leaf_label
    
    return tree

# Calculate the prediction error for a dataset
def calculate_error(tree, dataset, attributes):
    incorrect_predictions = 0
    for example in dataset:
        example_dict = {attr: val for attr, val in zip(attributes, example[:-1])}
        predicted = predict(tree, example_dict)
        actual = example[-1]
        if predicted != actual:
            incorrect_predictions += 1
    return incorrect_predictions / len(dataset)

# Predict the label for a given example using the decision stump
def predict(tree, example):
    if not isinstance(tree, dict):
        return tree  # Return the leaf node (class label)
    
    attribute = list(tree.keys())[0]
    subtree = tree[attribute].get(example[attribute], None)
    
    if subtree is None:
        return None  # Handle unseen values (if any)
    
    return subtree

# Function to run the experiment with decision stumps and metrics
def run_experiment(train_data, test_data, attributes, metrics, numerical_indices, handle_unknown_as_missing=True):
    results = []
    
    # Create a uniform weight for all examples initially
    initial_weights = np.ones(len(train_data)) / len(train_data)
    
    for metric in metrics:
        # Train the decision stump
        tree = id3_stump(train_data, attributes, initial_weights, metric=metric)
        
        # Calculate errors for both training and test sets
        train_error = calculate_error(tree, train_data, attributes)
        test_error = calculate_error(tree, test_data, attributes)
        
        results.append({
            'Metric': metric,
            'Train Error': train_error,
            'Test Error': test_error
        })
    
    return results

# Test function
def test_decision_stump():
    # A simple dataset with categorical and numerical attributes
    # Features: ['age', 'job', 'balance', 'loan']
    # Labels: 'yes' or 'no'
    train_data = [
        ['young', 'admin', 2000, 'no', 'no'],  # Feature columns + label
        ['young', 'admin', 3000, 'yes', 'no'],
        ['middle-aged', 'technician', 1500, 'no', 'no'],
        ['middle-aged', 'technician', 1000, 'no', 'no'],
        ['old', 'admin', 500, 'no', 'yes'],
        ['old', 'technician', 700, 'no', 'yes']
    ]

    test_data = [
        ['young', 'admin', 2500, 'yes', 'no'],  # Example to predict
        ['middle-aged', 'technician', 800, 'yes', 'yes'],
        ['old', 'admin', 600, 'no', 'yes']
    ]

    attributes = ['age', 'job', 'balance', 'loan']
    metrics = ['entropy']  # We can test with other metrics like 'gini', 'majority_error'
    numerical_indices = [2]  # 'balance' is a numerical attribute
    
    # Run experiment with stump decision trees
    experiment_results = run_experiment(train_data, test_data, attributes, metrics, numerical_indices, handle_unknown_as_missing=False)
    
    # Display results to verify the accuracy of predictions
    print("Test results:")
    print(experiment_results)

# Run the test
test_decision_stump()

def main():
    parser = argparse.ArgumentParser(description="Run ID3 decision stump experiment.")
    parser.add_argument('--train', required=True, help='Path to the training dataset')
    parser.add_argument('--test', required=True, help='Path to the test dataset')
    parser.add_argument('--metrics', nargs='+', default=['entropy', 'gini', 'majority_error'], help='Metrics to use for information gain (entropy, gini, majority_error)')
    parser.add_argument('--handle-unknown-as-missing', action='store_true', help='Fill missing values (unknown) with the majority value')
    
    args = parser.parse_args()
    
    # Load the datasets
    train_data = load_csv(args.train)
    test_data = load_csv(args.test)

    # Experiment parameters
    attributes = ['age', 'job', 'balance', 'loan']
    numerical_indices = [2]  # 'balance' is a numerical attribute
    
    # Run experiment (stump = max_depth of 1)
    results = run_experiment(
        train_data, 
        test_data, 
        attributes, 
        args.metrics, 
        numerical_indices, 
        handle_unknown_as_missing=args.handle_unknown_as_missing
    )
    
    # Output results
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()