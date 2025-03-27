import os
import csv
import numpy as np
import subprocess
import pytest
from LassoHomotopy.lasso_homotopy import LassoHomotopyModel

#  Works with column names like X1, X2, ..., X10
def load_csv(filename):
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    x_keys = [k for k in data[0].keys() if k.startswith("X")]
    X = np.array([[float(row[k]) for k in x_keys] for row in data])
    return X, data

#  Test 1 - small dataset
def test_small_dataset():
    path = "tests/small_test.csv"
    if not os.path.exists(path):
        pytest.skip("small_test.csv not found")

    X, raw = load_csv(path)
    y = np.array([float(row["y"]) for row in raw])

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)

    print("\n[TEST: Small Dataset]")
    print("Fitted Coefficients:", results.coef_)
    preds = results.predict(X)
    print("Predictions (first 5):", preds[:5])

    assert len(results.coef_) == X.shape[1]


#  Test 2 - collinear dataset (generate y inside)
def test_collinear_data_sparse_solution():
    path = "tests/collinear_data.csv"
    if not os.path.exists(path):
        pytest.skip("collinear_data.csv not found")

    X, raw = load_csv(path)
    x_keys = [k for k in raw[0] if k.startswith("X")]

    true_coef = np.array([3, 0, 0, 0, 5] + [0] * (len(x_keys) - 5))
    bias = 2
    noise = np.random.normal(0, 0.1, size=(X.shape[0], 1))
    y = (X @ true_coef.reshape(-1, 1) + bias + noise).ravel()

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)

    print("\n[TEST: Collinear Data]")
    print("Fitted Coefficients:", results.coef_)
    preds = results.predict(X)
    print("Predictions (first 5):", preds[:5])

    num_zeros = np.sum(np.abs(results.coef_) < 0.1)
    print("Zero Coefficients:", num_zeros)
    assert num_zeros >= 5


#  Test 3 - auto-generates data using your script
def test_generated_data_sparse_solution():
    path = "tests/generated_data.csv"

    result = subprocess.run(["python", "tests/generate_regression_data.py"], capture_output=True, text=True)
    print(result.stdout)
    assert result.returncode == 0, f"Generator failed:\n{result.stderr}"

    X, raw = load_csv(path)
    y = np.array([float(row["y"]) for row in raw])

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)

    print("\n[TEST: Generated Data]")
    print("Fitted Coefficients:", results.coef_)
    preds = results.predict(X)
    print("Predictions (first 5):", preds[:5])

    num_zeros = np.sum(np.abs(results.coef_) < 0.1)
    print("Zero Coefficients:", num_zeros)
    assert num_zeros >= 2

def test_irrelevant_feature():
    # Generate synthetic data
    np.random.seed(42)
    X_relevant = np.random.randn(100, 2)  # 2 useful features
    X_irrelevant = np.random.randn(100, 8)  # 8 noisy features
    X = np.hstack((X_relevant, X_irrelevant))

    true_coef = np.array([5, -3] + [0] * 8)
    y = X @ true_coef + np.random.normal(0, 0.1, 100)

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)

    print("\n[TEST: Irrelevant Feature]")
    print("Fitted Coefficients:", results.coef_)

    # Check if most of the irrelevant features are near zero
    num_zeros = np.sum(np.abs(results.coef_[2:]) < 0.1)
    print("Zero Coefficients in irrelevant features:", num_zeros)

    assert num_zeros >= 6  # At least 6 of 8 irrelevant features should be suppressed

