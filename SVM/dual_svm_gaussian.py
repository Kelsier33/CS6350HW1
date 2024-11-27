from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize
import pandas as pd


class SVMModelDualGaussian:
    def __init__(self, C, gamma):
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.X_support = None
        self.y_support = None
        self.b = None

    def gaussian_kernel(self, X1, X2):
        """Vectorized Gaussian kernel computation."""
        X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
        X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
        return np.exp(-(X1_sq - 2 * np.dot(X1, X2.T) + X2_sq) / self.gamma)

    def dual_objective(self, alpha, K, y):
        """Dual objective using precomputed kernel matrix."""
        return 0.5 * np.dot(alpha, np.dot(alpha * y, y * K)) - np.sum(alpha)

    def equality_constraint(self, alpha, y):
        return np.dot(alpha, y)

    def train(self, X, y):
        N = X.shape[0]
        K = self.gaussian_kernel(X, X)

        # Initial guess for alpha
        alpha_init = np.zeros(N)

        # Constraints
        constraints = [
            {"type": "eq", "fun": self.equality_constraint, "args": (y,)},
        ]
        bounds = [(0, self.C) for _ in range(N)]

        # Solve the dual problem
        result = minimize(
            fun=self.dual_objective,
            x0=alpha_init,
            args=(K, y),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.alpha = result.x

        # Support vectors
        self.support_vectors = np.where(self.alpha > 1e-5)[0]
        self.X_support = X[self.support_vectors]
        self.y_support = y[self.support_vectors]

        # Compute bias
        self.b = np.mean(
            [
                y_k - np.sum(
                    self.alpha[self.support_vectors]
                    * self.y_support
                    * K[k, self.support_vectors]
                )
                for k, y_k in zip(self.support_vectors, self.y_support)
            ]
        )

    def predict(self, X):
        """Predict using precomputed kernel values."""
        K = self.gaussian_kernel(X, self.X_support)
        return np.sign(np.dot(K, self.alpha[self.support_vectors] * self.y_support) + self.b)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        return errors / len(y) * 100



def train_and_evaluate(C, gamma, X_train, y_train, X_test, y_test):
    svm = SVMModelDualGaussian(C=C, gamma=gamma)
    svm.train(X_train, y_train)
    train_error = svm.evaluate(X_train, y_train)
    test_error = svm.evaluate(X_test, y_test)
    return C, gamma, train_error, test_error

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
gamma_values = [0.1, 0.5, 1, 5, 100]

# Perform parallelized training and evaluation
results = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate)(C, gamma, X_train, y_train, X_test, y_test)
    for C in C_values
    for gamma in gamma_values
)

# Summarize results
results_df = pd.DataFrame(
    results, columns=["C", "gamma", "Training Error (%)", "Test Error (%)"]
)

# Display the table in the terminal
print("\nSVM Gaussian Kernel Results:")
print(results_df.to_string(index=False))

def calculate_support_vector_overlap(C_values, gamma_values, X_train, y_train):
    """
    Calculate the number of support vectors for each (C, gamma) and 
    compute overlap between consecutive gamma values for C = 500/873.
    """
    support_vectors_results = {}  # Store support vectors for each (C, gamma)

    # Loop through C and gamma values
    for C in C_values:
        support_vectors_results[C] = {}
        for gamma in gamma_values:
            print(f"Training Dual SVM with C = {C} and gamma = {gamma}")
            svm = SVMModelDualGaussian(C=C, gamma=gamma)
            svm.train(X_train, y_train)
            support_vectors_results[C][gamma] = svm.support_vectors
            print(f"Number of support vectors: {len(svm.support_vectors)}\n")

    # Overlap computation for C = 500/873
    C_target = 500 / 873
    gamma_overlap_results = {}
    previous_support_vectors = None

    for gamma in gamma_values:
        current_support_vectors = support_vectors_results[C_target][gamma]
        if previous_support_vectors is not None:
            # Compute overlap
            overlap = len(np.intersect1d(previous_support_vectors, current_support_vectors))
            gamma_overlap_results[f"Overlap between γ={previous_gamma} and γ={gamma}"] = overlap
        previous_support_vectors = current_support_vectors
        previous_gamma = gamma

    return support_vectors_results, gamma_overlap_results


# Calculate support vector overlaps
support_vectors_results, gamma_overlap_results = calculate_support_vector_overlap(C_values, gamma_values, X_train, y_train)

# Display results
print("\nNumber of Support Vectors for Each (C, γ):")
for C, gamma_dict in support_vectors_results.items():
    print(f"\nC = {C}:")
    for gamma, support_vectors in gamma_dict.items():
        print(f"  γ = {gamma}: {len(support_vectors)} support vectors")

print("\nOverlaps Between Consecutive γ Values for C = 500/873:")
for key, overlap in gamma_overlap_results.items():
    print(f"{key}: {overlap} support vectors overlap")

