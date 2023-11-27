import numpy as np
from sklearn import datasets
from sklearn.svm import SVC

class ScratchSVMClassifier():
    def __init__(self, num_iter, lr,
                 kernel='linear',
                 threshold=1e-5,
                 verbose=False):

        self.iter = num_iter
        self.lr = lr
        self.kernel = kernel
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.lam_sv = np.zeros(n_samples)
        self.b = 0

        # Training loop
        for _ in range(self.iter):
            # Calculate gradient
            gradient = self._calculate_gradient(X, y)

            # Update Lagrange multipliers
            self.lam_sv -= self.lr * gradient

            # Clip values to be within [0, C]
            self.lam_sv = np.clip(self.lam_sv, 0, None)

            # Update bias term
            self.b = self._calculate_bias(X, y)

        # Select support vectors
        support_vector_indices = np.where(self.lam_sv >= self.threshold)[0]
        self.X_sv = X[support_vector_indices]
        self.y_sv = y[support_vector_indices]
        self.lam_sv = self.lam_sv[support_vector_indices]
        self.n_support_vectors = len(self.X_sv)

        if self.verbose:
            print(f"Number of support vectors: {self.n_support_vectors}")

    def _calculate_gradient(self, X, y):
        gradient = np.zeros_like(self.lam_sv)
        for i in range(len(self.lam_sv)):
            gradient[i] = 1 - y[i] * np.sum(self.lam_sv * y * self._kernel_function(X[i], X))
        return gradient

    def _calculate_bias(self, X, y):
        support_vector_indices = np.where(self.lam_sv > self.threshold)[0]
        if len(support_vector_indices) == 0:
            return 0  # No support vectors, set bias to 0

        bias_sum = 0
        for i in support_vector_indices:
            bias_sum += y[i] - np.sum(self.lam_sv * y * self._kernel_function(X[i, :], self.X_sv))

        return bias_sum / len(support_vector_indices)

    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            #x1 = np.expand_dims(x1, axis=1)
            return np.dot(x1, x2.T)
        # Add other kernel options here

    def _predict_one(self, x):
        if self.kernel == 'linear':
            return np.sum(self.lam_sv * self.y_sv * self._kernel_function(self.X_sv, x)) + self.b
        # Add other kernel options here

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            y_pred[i] = np.sign(self._predict_one(X[i]))

        return y_pred


### test ####
iris = datasets.load_iris()
X = iris['data']
Y = iris['target']
Y = np.where(Y>0, 1, 0)

n_iter = 100
lr = 0.001
bias = True
verbose=True

SVMClassifier = ScratchSVMClassifier(num_iter=10000,
                                     lr=0.1,
                                     threshold=0,
                                     verbose=verbose)

SVMClassifier.fit(X, Y)

y_pred = SVMClassifier.predict(X)
print(Y, y_pred)
