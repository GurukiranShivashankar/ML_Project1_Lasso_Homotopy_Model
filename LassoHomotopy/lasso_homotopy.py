import numpy as np

class LassoHomotopyModel():
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = y - y.mean()

        n_samples, n_features = X.shape
        coef_ = np.zeros(n_features)
        active_set = []
        inactive_set = list(range(n_features))
        residual = y.copy()

        for _ in range(self.max_iter):
            corr = X.T @ residual

            if len(inactive_set) == 0:
                break

            c_max = np.max(np.abs(corr[inactive_set]))
            if c_max < self.alpha:
                break

            j = inactive_set[np.argmax(np.abs(corr[inactive_set]))]
            active_set.append(j)
            inactive_set.remove(j)

            X_active = X[:, active_set]
            coef_active, _, _, _ = np.linalg.lstsq(X_active, y, rcond=None)

            coef_[:] = 0
            coef_[active_set] = coef_active
            residual = y - X @ coef_

            if np.linalg.norm(residual) < self.tol:
                break

        return LassoHomotopyResults(coef_)


class LassoHomotopyResults():
    def __init__(self, coef_):
        self.coef_ = coef_

    def predict(self, X):
        return X @ self.coef_
