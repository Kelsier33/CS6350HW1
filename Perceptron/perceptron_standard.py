import numpy as np
import pandas as pd

class PerceptronModel(object):
    def __init__(self, dimensions, learning_rate, epoches):
        """
        Initialize a new Perceptron instance.
        """
        # Initialize weights with zeros
        self.w = np.zeros(dimensions)
        self.learning_rate = learning_rate
        self.epoches = epoches

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        """
        return 1 if np.dot(self.w, x) >= 0 else 0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        steps:
        - shuffle the data
        - loop through each training example
            - if error, update the weights
        - return weights
        """
        np.random.shuffle(dataset)  # Shuffle dataset in-place
        for x, y in dataset:
            prediction = self.get_prediction(x)
            error = y - prediction
            if error != 0:
                self.w += error * x * self.learning_rate  # Update weights for misclassified points
        return self.w

# ========== Data Loading and Model Setup ==========
learning_rate = .78

# Load training and testing datasets
train_data = pd.read_csv('bank-note/bank-note/train.csv')
test_data = pd.read_csv('bank-note/bank-note/test.csv')

# Separate features and labels
X_train = train_data.iloc[:, :-1].values  # Features of training data
y_train = train_data.iloc[:, -1].values   # Labels of training data
X_test = test_data.iloc[:, :-1].values    # Features of test data
y_test = test_data.iloc[:, -1].values     # Labels of test data

# Initialize Perceptron model with the number of features
dimensions = X_train.shape[1]
model = PerceptronModel(dimensions, learning_rate, 10)

# Combine features and labels into a single dataset for training
training_dataset = list(zip(X_train, y_train))

# Train the model
for epoch in range(10):
    model.train(training_dataset)
def runTest():
    error = 0
    success = 0
    for x, y in zip(X_test, y_test):
        prediction = model.get_prediction(x)
        if prediction != y:
            error += 1
            continue
        success += 1
    return error, success

error, success = runTest()
print(error, success)

# Calculate the average error as a percentage
total_predictions = error + success
average_error_percentage = (error / total_predictions) * 100

print("Average Prediction Error: {:.2f}%".format(average_error_percentage))
    
# Print the final weights after training
print("Trained weights:", model.get_weights())
