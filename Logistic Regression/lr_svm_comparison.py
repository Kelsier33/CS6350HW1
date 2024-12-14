from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data(train_path, test_path):
    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data("train.csv", "test.csv")
# Train SVM model
svm_model = SVC(kernel="linear", C=1)  # Linear kernel for binary classification
svm_model.fit(X_train, y_train)
svm_train_predictions = svm_model.predict(X_train)
svm_test_predictions = svm_model.predict(X_test)
svm_train_error = 1 - accuracy_score(y_train, svm_train_predictions)
svm_test_error = 1 - accuracy_score(y_test, svm_test_predictions)

# Train Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_train_predictions = logreg_model.predict(X_train)
logreg_test_predictions = logreg_model.predict(X_test)
logreg_train_error = 1 - accuracy_score(y_train, logreg_train_predictions)
logreg_test_error = 1 - accuracy_score(y_test, logreg_test_predictions)



# FNN Results from Image
fnn_train_error_random = {
    5: 0.0103,
    10: 0.0034,
    25: 0.0103,
    50: 0.0000,
    100: 0.0000,
}

fnn_test_error_random = {
    5: 0.0120,
    10: 0.0040,
    25: 0.0120,
    50: 0.0000,
    100: 0.0000,
}

fnn_train_error_zero = {
    5: 0.0252,
    10: 0.0092,
    25: 0.0241,
    50: 0.0195,
    100: 0.0115,
}

fnn_test_error_zero = {
    5: 0.0200,
    10: 0.0080,
    25: 0.0220,
    50: 0.0180,
    100: 0.0100,
}

# Combine Results
results = {
    "Model": [],
    "Width": [],
    "Training Error": [],
    "Test Error": [],
}

# Add FNN (Random Initialization)
for width, train_error in fnn_train_error_random.items():
    results["Model"].append("FNN (Random Init)")
    results["Width"].append(width)
    results["Training Error"].append(train_error)
    results["Test Error"].append(fnn_test_error_random[width])

# Add FNN (Zero Initialization)
for width, train_error in fnn_train_error_zero.items():
    results["Model"].append("FNN (Zero Init)")
    results["Width"].append(width)
    results["Training Error"].append(train_error)
    results["Test Error"].append(fnn_test_error_zero[width])

# Add SVM and Logistic Regression
results["Model"].extend(["SVM", "Logistic Regression"])
results["Width"].extend(["N/A", "N/A"])
results["Training Error"].extend([svm_train_error, logreg_train_error])
results["Test Error"].extend([svm_test_error, logreg_test_error])

# Create DataFrame
import pandas as pd
results_df = pd.DataFrame(results)

# Display results
print(results_df)