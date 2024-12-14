import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layers = [input_size, hidden_size, hidden_size, output_size]
        self.weights = [
            np.zeros((self.layers[i + 1], self.layers[i] + 1))
            for i in range(len(self.layers) - 1)
        ]
        self.weights = [
            np.random.randn(self.layers[i + 1], self.layers[i] + 1) 
            for i in range(len(self.layers) - 1)
        ]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for weight in self.weights:
            X = np.c_[np.ones(X.shape[0]), X]  # Add bias
            z = np.dot(X, weight.T)
            self.z_values.append(z)
            X = self.sigmoid(z)
            self.activations.append(X)
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        y = y.reshape(-1, 1)
        deltas = []

        # Output layer error
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.z_values[-1])
        deltas.append(delta)

        # Hidden layers
        for l in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[l][:, 1:]) * self.sigmoid_derivative(
                self.z_values[l - 1]
            )
            deltas.append(delta)

        deltas.reverse()

        # Gradients
        gradients = []
        for l in range(len(self.weights)):
            a = np.c_[np.ones(self.activations[l].shape[0]), self.activations[l]]
            grad = np.dot(deltas[l].T, a) / m
            gradients.append(grad)

        return gradients

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def compute_error(self, X, y):
        predictions = self.forward(X)
        predictions = (predictions > 0.5).astype(int)
        return np.mean(predictions != y.reshape(-1, 1))


def load_data(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test


def train_and_evaluate(hidden_size, X_train, y_train, X_test, y_test, gamma_0, d, epochs):
    input_size = X_train.shape[1]
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    t = 0
    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        # SGD Loop
        for i in range(len(X_train)):
            x = X_train[i].reshape(1, -1)
            y = y_train[i].reshape(1, -1)
            learning_rate = gamma_0 / (1 + t*gamma_0/ d)
            t += 1

            # Forward and Backward passes
            nn.forward(x)
            gradients = nn.backward(x, y)

            # Update weights
            nn.update_weights(gradients, learning_rate)

    # Compute errors
    train_error = nn.compute_error(X_train, y_train)
    test_error = nn.compute_error(X_test, y_test)

    return train_error, test_error


if __name__ == "__main__":
    # Load dataset
    X_train, y_train, X_test, y_test = load_data("train.csv", "test.csv")

    # Hyperparameters
    gamma_0 = 1
    d = 1000
    epochs = 1
    widths = [5, 10, 25, 50, 100]

    # Train and evaluate for different hidden layer widths
    results = []
    for width in widths:
        train_error, test_error = train_and_evaluate(
            width, X_train, y_train, X_test, y_test, gamma_0, d, epochs
        )
        results.append((width, train_error, test_error))
        print(f"Width: {width}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")