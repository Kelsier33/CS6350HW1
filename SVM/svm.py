import numpy as np
import pandas as pd

class SVMModel(object):
    def __init__(self, dimensions, C, learning_rate_schedule, epochs):
        """
        Initialize a new SVM instance.
        """
        self.w = np.zeros(dimensions)  # Initialize weights with zeros
        self.b = 0  # Initialize bias with 0
        self.C = C  # Regularization parameter
        self.learning_rate_schedule = learning_rate_schedule  # Learning rate schedule function
        self.epochs = epochs  # Number of training epochs

    def get_weights(self):
        """
        Return the weights and bias of the model.
        """
        return self.w, self.b

    def predict(self, x):
        """
        Predict the class for a single data point `x`.
        """
        return 1 if np.dot(self.w, x) + self.b >= 0 else -1

    def train(self, X, y):
        """
        Train the SVM using the stochastic sub-gradient descent algorithm.
        """
        n = len(y)
        iteration = 1  # Global iteration counter for the learning rate schedule
        for epoch in range(self.epochs):
            for i in range(n):
                # Compute the learning rate for the current iteration
                eta_t = self.learning_rate_schedule(iteration, self.C)
                iteration += 1

                # Hinge loss update
                if y[i] * (np.dot(self.w, X[i]) + self.b) < 1:
                    self.w = (1 - eta_t) * self.w + eta_t * self.C * y[i] * X[i]
                    self.b += eta_t * self.C * y[i]
                else:
                    # Regularization update
                    self.w = (1 - eta_t) * self.w
        return self.w, self.b

    def evaluate(self, X, y):
        """
        Evaluate the model on a dataset.
        Returns the number of errors and the total number of predictions.
        """
        errors = 0
        for i in range(len(y)):
            prediction = self.predict(X[i])
            if prediction != y[i]:
                errors += 1
        return errors, len(y)

# Define learning rate schedules
def first_schedule(iteration, C, initial_learning_rate=0.01):
    """
    First learning rate schedule: eta_t = eta_0 / (1 + eta_0 * t / C)
    """
    return initial_learning_rate / (1 + (initial_learning_rate * (iteration - 1)) / C)

def second_schedule(iteration, C, initial_learning_rate=0.01):
    """
    Second learning rate schedule: eta_t = eta_0 / (1 + t)
    """
    return initial_learning_rate / (1 + (iteration - 1))

# File paths for training and test datasets
train_file = "train.csv"  # Replace with the path to your training file
test_file = "test.csv"    # Replace with the path to your test file

# Load the CSV files
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Assume the last column is the label
label_column = train_data.columns[-1]

# Convert labels to {1, -1}
train_data[label_column] = train_data[label_column].apply(lambda x: 1 if x == 1 else -1)
test_data[label_column] = test_data[label_column].apply(lambda x: 1 if x == 1 else -1)

# Separate features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# SVM hyperparameters
C_values = [100/873, 500/873, 700/873]
epochs = 100

# Choose the learning rate schedule function
learning_rate_schedule = first_schedule  
#learning_rate_schedule = second_schedule

# Train and evaluate the SVM for each C
for C in C_values:
    print(f"Training with C = {C}")
    svm = SVMModel(dimensions=X_train.shape[1], C=C, learning_rate_schedule=learning_rate_schedule, epochs=epochs)
    svm.train(X_train, y_train)
    train_errors, train_total = svm.evaluate(X_train, y_train)
    test_errors, test_total = svm.evaluate(X_test, y_test)

    # Print results
    print("Weights: ", svm.get_weights())
    print(f"Training error: {train_errors}/{train_total} ({(train_errors / train_total) * 100:.2f}%)")
    print(f"Test error: {test_errors}/{test_total} ({(test_errors / test_total) * 100:.2f}%)\n")


