import numpy as np
import csv
import os

def load_X_from_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    x_keys = [k for k in data[0].keys() if k.startswith("X")]
    X = np.array([[float(row[k]) for k in x_keys] for row in data])
    return X

def write_X_y_to_csv(X, y, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [f"X{i+1}" for i in range(X.shape[1])] + ["y"]
        writer.writerow(headers)
        for i in range(X.shape[0]):
            row = list(X[i]) + [y[i]]
            writer.writerow(row)

def main():
    input_path = "tests/collinear_data.csv"
    output_path = "tests/generated_data.csv"

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print(f"[INFO] Using feature data from: {input_path}")

    X = load_X_from_csv(input_path)

    true_coef = np.array([3, 0, 0, 0, 5] + [0] * (X.shape[1] - 5))
    bias = 2
    noise = np.random.normal(0, 0.1, size=(X.shape[0], 1))
    y = (X @ true_coef.reshape(-1, 1) + bias + noise).ravel()

    write_X_y_to_csv(X, y, output_path)
    print(f"[SUCCESS] Generated data saved to: {output_path}")

if __name__ == "__main__":
    main()
