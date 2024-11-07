import numpy as np
import pandas as pd

class VotedPerceptronModel(object):
    def __init__(self, dimensions, learning_rate, epoches):
        """
        Initialize a new Perceptron instance.
        """
        # Initialize weights with zeros
        self.w = [(np.zeros(dimensions), 1)]
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
        sum = 0
        for w_vec, count in self.w:
            sum += count * sign(np.dot(w_vec, x))
        return 1 if  sign(sum) >= 0 else 0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        steps:
        - shuffle the data
        - loop through each training example
            - if error, update the weights
        - return weights
        """
        for epoch in range(self.epoches):
            np.random.shuffle(dataset)  # Shuffle the dataset each epoch
            for x, y in dataset:
                prediction = self.get_prediction(x)
                error = y - prediction
                if error != 0:
                     # Get the current weight vector and create a new one with the update
                    new_vec = self.w[-1][0] + error * x * self.learning_rate  
                    # Append the new weight vector with a count of 1
                    self.w.append((new_vec, 1))
                else:
                    # Increment the count of the current weight vector
                    self.w[-1] = (self.w[-1][0], self.w[-1][1] + 1)
        return self.w
    
def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if(x >= 0):
        return 1
    else:
        return -1
        


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
model = VotedPerceptronModel(dimensions, learning_rate, 10)

# Combine features and labels into a single dataset for training
training_dataset = list(zip(X_train, y_train))

# Train the model
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
print("Trained weights:", model.get_weights()[-1])
