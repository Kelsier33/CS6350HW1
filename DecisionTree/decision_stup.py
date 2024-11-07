import pandas as pd
import numpy as np

# Load the train and test datasets directly
train_df = pd.read_csv('train_.csv')
test_df = pd.read_csv('test_.csv')

# Identify the last column in both datasets as the target column
target_column_train = train_df.columns[-1]
target_column_test = test_df.columns[-1]

# Convert the last column to binary (1 for 'yes', 0 for 'no')
train_df['y'] = train_df[target_column_train].apply(lambda x: 1 if x == 'yes' else 0)
test_df['y'] = test_df[target_column_test].apply(lambda x: 1 if x == 'yes' else 0)

# Drop the original last column (as it's now converted to 'y')
train_df = train_df.drop(target_column_train, axis=1)
test_df = test_df.drop(target_column_test, axis=1)

# Preprocessing: Convert categorical columns to numerical (using one-hot encoding)
train_df_encoded = pd.get_dummies(train_df, drop_first=True)  # Drop first to avoid multicollinearity
test_df_encoded = pd.get_dummies(test_df, drop_first=True)    # Same encoding for test set

# Separate features (X) and target (y)
X_train = train_df_encoded.drop('y', axis=1)  # Drop target column from training features
y_train = train_df_encoded['y'].values  # Convert target to numpy array

X_test = test_df_encoded.drop('y', axis=1)  # Drop target column from test features
y_test = test_df_encoded['y'].values  # Convert target to numpy array

# Align columns to ensure both train and test sets have the same columns
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# Function to calculate accuracy (replacing `accuracy_score`)
def accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

# Decision Stump Implementation
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y):
        """
        Fit the decision stump based on the training data X and labels y.
        """
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Iterate through all features to find the best split
        for feature_index in range(n_features):
            X_column = X.iloc[:, feature_index]  # Get feature column
            thresholds = np.unique(X_column)  # Possible split thresholds

            # Try splitting with each threshold
            for threshold in thresholds:
                predictions = np.ones(n_samples)
                predictions[X_column < threshold] = -1  # Classify based on threshold

                # Calculate error
                error = sum(y != predictions)

                # Adjust polarity if error is more than half
                if error > n_samples / 2:
                    error = n_samples - error
                    polarity = -1
                else:
                    polarity = 1

                # Update minimum error and best parameters
                if error < min_error:
                    min_error = error
                    self.polarity = polarity
                    self.threshold = threshold
                    self.feature_index = feature_index

    def predict(self, X):
        """
        Predict the class labels for the given samples X.
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X.iloc[:, self.feature_index]

        # Apply the threshold and polarity for prediction
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values > self.threshold] = -1

        return predictions

# Initialize and train the decision stump
stump = DecisionStump()
stump.fit(X_train, y_train)

# Predict on the test data
predictions = stump.predict(X_test)

# Calculate accuracy (without `sklearn`)
accuracy = accuracy_score(y_test, predictions)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
