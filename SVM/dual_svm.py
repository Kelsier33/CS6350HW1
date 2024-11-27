import numpy as np
from scipy.optimize import minimize
import pandas as pd

class SVMModelDual:
    def __init__(self, C):
        """
        Initialize the SVMModelDual.
        Args:
            C (float): Regularization parameter.
        """
        self.C = C  # Regularization parameter
        self.alpha = None  # Lagrange multipliers
        self.w = None  # Weights
        self.b = None  # Bias
        self.support_vectors = None  # Indices of support vectors

    @staticmethod
    def linear_kernel(x1, x2):
        """Linear kernel for dot product."""
        return np.dot(x1, x2)

    def dual_objective(self, alpha, X, y):
        """
        Optimized dual objective function using matrix operations.
        """
        # Compute the kernel matrix (linear kernel: X @ X.T)
        K = np.dot(X, X.T)

        # Compute the dual objective function
        return 0.5 * np.dot(alpha, np.dot(alpha * y, y * K)) - np.sum(alpha)

    def equality_constraint(self, alpha, y):
        """Equality constraint: sum(alpha_i * y_i) = 0."""
        return np.dot(alpha, y)

    def train(self, X, y):
        """
        Train SVM in the dual domain.
        Args:
            X (np.ndarray): Feature matrix (N x d).
            y (np.ndarray): Labels (N x 1).
        """
        N = X.shape[0]

        # Initial guess for alpha
        alpha_init = np.zeros(N)

        # Constraints
        constraints = [
            {"type": "eq", "fun": self.equality_constraint, "args": (y,)},
        ]
        bounds = [(0, self.C) for _ in range(N)]

        # Solve dual problem
        result = minimize(
            fun=self.dual_objective,
            x0=alpha_init,
            args=(X, y),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Extract optimized alphas
        self.alpha = result.x

        # Support vectors
        self.support_vectors = np.where(self.alpha > 1e-5)[0]

        # Compute weights (w = sum(alpha_i * y_i * x_i))
        self.w = np.sum(self.alpha[:, None] * y[:, None] * X, axis=0)

        # Compute bias (b = y_k - w^T x_k for any support vector k)
        self.b = y[self.support_vectors[0]] - np.dot(self.w, X[self.support_vectors[0]])

    def predict(self, X):
        """
        Predict labels for the input data.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted labels.
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def evaluate(self, X, y):
        """
        Evaluate the model on a dataset.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True labels.
        Returns:
            float: Error rate (percentage).
        """
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        return errors / len(y) * 100

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

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]

for C in C_values:
    print(f"Training Dual SVM with C = {C}")

    # Train the SVM
    svm = SVMModelDual(C=C)
    svm.train(X_train, y_train)

    # Evaluate on train and test data
    train_error = svm.evaluate(X_train, y_train)
    test_error = svm.evaluate(X_test, y_test)

    # Print results
    print(f"Weights: {svm.w}")
    print(f"Bias: {svm.b}")
    print(f"Number of support vectors: {len(svm.support_vectors)}")
    print(f"Training error: {train_error:.2f}%")
    print(f"Test error: {test_error:.2f}%\n")
