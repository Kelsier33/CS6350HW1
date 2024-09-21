import math
import csv
from collections import Counter
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run ID3 decision tree experiment.")
    parser.add_argument('--train', required=True, help='Path to the training dataset')
    parser.add_argument('--test', required=True, help='Path to the test dataset')
    parser.add_argument('--max-depth', type=int, default=6, help='Maximum depth of the decision tree')
    parser.add_argument('--metrics', nargs='+', default=['entropy', 'gini', 'majority_error'], help='Metrics to use for information gain (entropy, gini, majority_error)')
    parser.add_argument('--handle-unknown-as-missing', action='store_true', help='Fill missing values (unknown) with the majority value')
    parser.add_argument('--handle-unknown-as-valid', action='store_true', help='Treat "unknown" as a valid value (no filling)')
    
    args = parser.parse_args()
    
    # Load the datasets
    train_data = load_csv(args.train)
    test_data = load_csv(args.test)

    # Experiment parameters
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    numerical_indices = [0, 5, 9, 11, 12, 13, 14]  # Adjust based on the dataset description
    
    # Run experiment
    results = run_experiment(
        train_data, 
        test_data, 
        attributes, 
        list(range(1, args.max_depth + 1)), 
        args.metrics, 
        numerical_indices, 
        handle_unknown_as_missing=args.handle_unknown_as_missing
    )
    
    # Output results (you can improve this part based on your needs)
    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()


# Function to calculate the entropy of a dataset
def entropy(subset):
    total = len(subset)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    entropy_value = 0
    for count in label_counts.values():
        prob = count / total
        entropy_value -= prob * math.log2(prob)
    
    return entropy_value

# Function to calculate the Gini Index of a dataset
def gini_index(subset):
    total = len(subset)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    gini = 1.0
    for count in label_counts.values():
        prob = count / total
        gini -= prob ** 2
    
    return gini

# Function to calculate the Majority Error of a dataset
def majority_error(subset):
    total = len(subset)
    if total == 0:
        return 0
    
    label_counts = Counter(row[-1] for row in subset)
    majority_class_count = max(label_counts.values())
    majority_class_proportion = majority_class_count / total
    
    return 1 - majority_class_proportion

# Function to calculate information gain based on the selected metric
def information_gain(dataset, attribute_index, metric='entropy'):
    if metric == 'entropy':
        total_metric = entropy(dataset)
    elif metric == 'gini':
        total_metric = gini_index(dataset)
    elif metric == 'majority_error':
        total_metric = majority_error(dataset)
    
    total_size = len(dataset)
    attribute_values = set(row[attribute_index] for row in dataset)
    
    weighted_metric = 0
    for value in attribute_values:
        subset = [row for row in dataset if row[attribute_index] == value]
        weight = len(subset) / total_size
        if metric == 'entropy':
            weighted_metric += weight * entropy(subset)
        elif metric == 'gini':
            weighted_metric += weight * gini_index(subset)
        elif metric == 'majority_error':
            weighted_metric += weight * majority_error(subset)
    
    return total_metric - weighted_metric

# Function to compute median for numerical attributes
def compute_medians(dataset, numerical_indices):
    medians = {}
    for idx in numerical_indices:
        values = sorted(float(row[idx]) for row in dataset if row[idx] != 'unknown')
        medians[idx] = values[len(values) // 2]  # Median
    return medians

# Function to binarize numerical features based on the median threshold
def binarize_numerical_features(dataset, medians):
    binarized_data = []
    for row in dataset:
        new_row = row[:]
        for idx, median in medians.items():
            if row[idx] != 'unknown':
                new_row[idx] = 'greater' if float(row[idx]) > median else 'less'
        binarized_data.append(new_row)
    return binarized_data

# Function to handle "unknown" values by filling them with the majority value of the attribute
def fill_missing_values_with_majority(dataset, attributes):
    filled_data = []
    majority_values = {}
    
    for i, attribute in enumerate(attributes):
        # Find the majority value for this attribute
        counts = Counter(row[i] for row in dataset if row[i] != 'unknown')
        majority_value = counts.most_common(1)[0][0]
        majority_values[attribute] = majority_value
    
    for row in dataset:
        new_row = row[:]
        for i, attribute in enumerate(attributes):
            if new_row[i] == 'unknown':
                new_row[i] = majority_values[attribute]  # Replace "unknown" with majority value
        filled_data.append(new_row)
    
    return filled_data

# ID3 algorithm with numerical feature support
def id3(dataset, attributes, metric='entropy', max_depth=None, depth=0):
    labels = [row[-1] for row in dataset]
    if len(set(labels)) == 1:
        return labels[0]
    
    if max_depth is not None and depth == max_depth:
        return Counter(labels).most_common(1)[0][0]
    
    best_attribute_index = max(range(len(attributes)), key=lambda i: information_gain(dataset, i, metric))
    tree = {attributes[best_attribute_index]: {}}
    new_attributes = attributes[:best_attribute_index] + attributes[best_attribute_index+1:]
    
    attribute_values = set(row[best_attribute_index] for row in dataset)
    for value in attribute_values:
        subset = [row for row in dataset if row[best_attribute_index] == value]
        subtree = id3(subset, new_attributes, metric, max_depth, depth + 1)
        tree[attributes[best_attribute_index]][value] = subtree
    
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

# Predict the label for a given example
def predict(tree, example):
    if not isinstance(tree, dict):
        return tree  # Return the leaf node (class label)
    
    attribute = list(tree.keys())[0]
    subtree = tree[attribute].get(example[attribute], None)
    
    if subtree is None:
        return None  # Handle unseen values (if any)
    
    return predict(subtree, example)

# Function to run the experiment with varying maximum depths and metrics
def run_experiment(train_data, test_data, attributes, max_depth_values, metrics, numerical_indices, handle_unknown_as_missing=True):
    results = []
    
    # Compute medians for numerical features in the training set
    medians = compute_medians(train_data, numerical_indices)
    
    # Optionally fill missing values in train and test datasets
    if handle_unknown_as_missing:
        train_data = fill_missing_values_with_majority(train_data, attributes)
        test_data = fill_missing_values_with_majority(test_data, attributes)
    
    # Binarize numerical features in both train and test data
    binarized_train_data = binarize_numerical_features(train_data, medians)
    binarized_test_data = binarize_numerical_features(test_data, medians)
    
    for metric in metrics:
        for max_depth in max_depth_values:
            # Train the decision tree using the given metric and max depth
            tree = id3(binarized_train_data, attributes, metric=metric, max_depth=max_depth)
            
            # Calculate errors for both training and test sets
            train_error = calculate_error(tree, binarized_train_data, attributes)
            test_error = calculate_error(tree, binarized_test_data, attributes)
            
            results.append({
                'Metric': metric,
                'Max Depth': max_depth,
                'Train Error': train_error,
                'Test Error': test_error
            })
    
    return results

# Load dataset from a CSV file
def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data[1:]  # Skip header row

# Load the training and test datasets
train_data = load_csv('train_.csv')
test_data = load_csv('test_.csv')

# Attributes and numerical attribute indices based on the dataset description
attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
numerical_indices = [0, 5, 9, 11, 12, 13, 14]  # Indices of numerical attributes

# Experiment parameters
max_depth_values = list(range(1, 17))
metrics = ['entropy', 'gini', 'majority_error']

# Run the experiment with "unknown" treated as missing
experiment_results = run_experiment(train_data, test_data, attributes, max_depth_values, metrics, numerical_indices, handle_unknown_as_missing=True)

# Print experiment results
import pandas as pd
df = pd.DataFrame(experiment_results)
print(df)
