import math
import numpy as np
import pandas as pd
import argparse
import csv
from collections import Counter
import matplotlib.pyplot as plt

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

# Function to calculate information gain based on the selected metric
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

# Load dataset from a CSV file
def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data[1:]  # Skip header row

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

# Predict the label for a given example using the decision stump
def predict(tree, example):
    if not isinstance(tree, dict):
        return tree  # Return the leaf node (class label)
    
    attribute = list(tree.keys())[0]
    subtree = tree[attribute].get(example[attribute], None)
    
    if subtree is None:
        return None  # Handle unseen values (if any)
    
    return subtree

# AdaBoost algorithm using decision stumps with efficient error and weight calculation
def adaboost(train_data, test_data, attributes, T):
    m = len(train_data)
    weights = np.ones(m) / m  # Initialize uniform weights
    stumps = []
    alphas = []
    
    for t in range(T):
        # Train a decision stump with current weights
        stump = id3_stump(train_data, attributes, weights)
        
        # Calculate the weighted error
        error_t = calculate_weighted_error(stump, train_data, weights, attributes)
        
        # Compute the stump's weight (alpha)
        alpha_t = 0.5 * np.log((1 - error_t) / (error_t + 1e-10))  # Add a small epsilon to avoid division by zero
        
        # Store the stump and its weight
        stumps.append(stump)
        alphas.append(alpha_t)
        
        # Update weights
        weights = update_weights(weights, alpha_t, stump, train_data, attributes)
    
    return stumps, alphas

# Function to calculate the weighted error
def calculate_weighted_error(stump, dataset, weights, attributes):
    predictions = np.array([predict(stump, {attr: val for attr, val in zip(attributes, example[:-1])}) for example in dataset])
    actuals = np.array([example[-1] for example in dataset])
    
    # Binary classification with 'yes' and 'no', convert to 1 and -1
    actuals_binary = np.where(actuals == 'yes', 1, -1)
    predictions_binary = np.where(predictions == 'yes', 1, -1)
    
    errors = np.where(predictions_binary != actuals_binary, 1, 0)
    weighted_error = np.sum(weights * errors) / np.sum(weights)
    
    return weighted_error

# Efficient weight updates
def update_weights(weights, alpha, stump, dataset, attributes):
    predictions = np.array([predict(stump, {attr: val for attr, val in zip(attributes, example[:-1])}) for example in dataset])
    actuals = np.array([example[-1] for example in dataset])
    
    actuals_binary = np.where(actuals == 'yes', 1, -1)
    predictions_binary = np.where(predictions == 'yes', 1, -1)
    
    # Update the weights: Misclassified examples get increased weights
    exponent = -alpha * actuals_binary * predictions_binary
    new_weights = weights * np.exp(exponent)
    return new_weights / np.sum(new_weights)  # Normalize weights

# Function to binarize numerical features based on the median value
def binarize_numerical_features(dataset, numerical_columns):
    binarized_data = dataset.copy()
    for col in numerical_columns:
        median_value = binarized_data[col].median()
        binarized_data[col] = binarized_data[col].apply(lambda x: 'greater' if x > median_value else 'less')
    return binarized_data

# Final prediction based on weighted majority of all stumps
def adaboost_predict(stumps, alphas, example, attributes):
    total_prediction = 0
    for stump, alpha in zip(stumps, alphas):
        predicted = predict(stump, example)
        total_prediction += alpha * (1 if predicted == 'yes' else -1)  # Assuming binary classification 'yes' and 'no'
    
    return 'yes' if total_prediction > 0 else 'no'

# Function to calculate the prediction error for the entire dataset using AdaBoost
def calculate_adaboost_error(stumps, alphas, dataset, attributes):
    incorrect_predictions = 0
    for example in dataset:
        example_dict = {attr: val for attr, val in zip(attributes, example[:-1])}
        predicted = adaboost_predict(stumps, alphas, example_dict, attributes)
        actual = example[-1]
        if predicted != actual:
            incorrect_predictions += 1
    
    return incorrect_predictions / len(dataset)

# Run the AdaBoost experiment with debugging
def run_adaboost_experiment_with_debug(train_data, test_data, attributes, max_T):
    results = []
    stump_train_errors = []
    stump_test_errors = []
    
    for T in range(1, max_T + 1):
        print(f"Iteration {T}:")
        
        # Train AdaBoost with T iterations
        stumps, alphas = adaboost(train_data, test_data, attributes, T)
        
        # Calculate training and test errors
        train_error = calculate_adaboost_error(stumps, alphas, train_data, attributes)
        test_error = calculate_adaboost_error(stumps, alphas, test_data, attributes)
        
        print(f"  Train Error after {T} iterations: {train_error}")
        print(f"  Test Error after {T} iterations: {test_error}")
        
        results.append({
            'Iterations': T,
            'Train Error': train_error,
            'Test Error': test_error
        })
        
        # Calculate individual stump errors for this iteration
        stump_train_error = calculate_weighted_error(stumps[-1], train_data, np.ones(len(train_data)) / len(train_data), attributes)
        stump_test_error = calculate_weighted_error(stumps[-1], test_data, np.ones(len(test_data)) / len(test_data), attributes)
        
        print(f"  Stump Train Error for iteration {T}: {stump_train_error}")
        print(f"  Stump Test Error for iteration {T}: {stump_test_error}")
        
        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)
    
    return pd.DataFrame(results), stump_train_errors, stump_test_errors

# Plotting functions
def plot_errors_over_iterations(results):
    plt.figure(figsize=(10, 6))
    
    # Plot training and test errors over iterations
    plt.plot(results['Iterations'], results['Train Error'], label='Train Error', marker='o')
    plt.plot(results['Iterations'], results['Test Error'], label='Test Error', marker='o')
    
    plt.title('Training and Test Errors vs Number of Iterations (T)')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stump_errors_over_iterations(results, stump_train_errors, stump_test_errors):
    plt.figure(figsize=(10, 6))
    
    # Plot training and test errors for individual decision stumps in each iteration
    plt.plot(results['Iterations'], stump_train_errors, label='Stump Train Error', marker='o')
    plt.plot(results['Iterations'], stump_test_errors, label='Stump Test Error', marker='o')
    
    plt.title('Stump Training and Test Errors vs Number of Iterations (T)')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to run the AdaBoost experiment
def main():
    parser = argparse.ArgumentParser(description="Run AdaBoost with decision stumps.")
    parser.add_argument('--train', required=True, help='Path to the training dataset')
    parser.add_argument('--test', required=True, help='Path to the test dataset')
    parser.add_argument('--max-T', type=int, default=500, help='Maximum number of boosting iterations (T)')
    
    args = parser.parse_args()
    
    # Load datasets
    train_data = load_csv(args.train)
    test_data = load_csv(args.test)

    # Define attributes (from CSV headers, adjust as necessary)
    attributes = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    # Apply binarization or any other preprocessing to the dataset if needed (already assumed in previous steps)
    
    # Run AdaBoost experiment with debugging for analysis
    max_T = args.max_T
    results, stump_train_errors, stump_test_errors = run_adaboost_experiment_with_debug(
        train_data, test_data, attributes, max_T
    )
    
    # Plot performance results
    plot_errors_over_iterations(results)
    plot_stump_errors_over_iterations(results, stump_train_errors, stump_test_errors)

if __name__ == "__main__":
    main()
