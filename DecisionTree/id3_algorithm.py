import math
import csv
from collections import Counter, defaultdict

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

# ID3 algorithm
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

# Load dataset from a CSV file
def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data[1:]  # Skip header row

# Load the training data
train_data = load_csv('/mnt/data/train.csv')

# Attributes from the data description file (without the label)
attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Running the ID3 algorithm with entropy and max depth of 3
decision_tree = id3(train_data, attributes, metric='entropy', max_depth=3)

# Print the generated decision tree
print(decision_tree)
