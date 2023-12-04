import numpy as np
import matplotlib.pyplot as plt

def MSE(y_pred, y):
    mse = np.mean((y - y_pred) ** 2)
    return mse

class ScratchLogisticRegression():
    def __init__(self, num_iter, lr, alpha, no_bias, verbose):
        self.iter = num_iter
        self.lr = lr
        self.alpha = alpha  # Regularization parameter
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        if not self.no_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            if X_val is not None:
                X_val = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1)

        self.coef_ = np.random.random((X.shape[1],))

        for epoch in range(self.iter):
            y_pred = self._logit_hypothesis(X)
            self.loss[epoch] = np.mean((-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))) + \
                               (self.alpha / 2) * np.mean(self.coef_[1:] ** 2)  # Regularization term added
            self._gradient_descent(X, y, y_pred)

            if X_val is not None:
                y_pred_val = self._logit_hypothesis(X_val)
                self.val_loss[epoch] = np.mean((-y_val * np.log(y_pred_val) - (1 - y_val) * np.log(1 - y_pred_val))) + \
                                       (self.alpha / 2) * np.mean(self.coef_[1:] ** 2)

            if self.verbose:
                if X_val is not None:
                    print(f"Epoch-{epoch}: train loss={self.loss[epoch]}, val loss={self.val_loss[epoch]}")
                else:
                    print(f"Epoch-{epoch}: train loss={self.loss[epoch]}")

        return self.coef_, self.loss, self.val_loss

    def predict(self, X):
        if not self.no_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        prob = self._logit_hypothesis(X)
        y_pred = np.where(prob >= 0.5, 1, 0)
        return y_pred

    def predict_proba(self, X):
        if not self.no_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        prob = self._logit_hypothesis(X)
        return prob

    def _logit_hypothesis(self, X):
        z = np.dot(X, self.coef_)
        y_pred = self.sigmoid(z)
        return y_pred

    def sigmoid(self, z):
        prob = 1 / (1 + np.exp(-z))
        return prob

    def _gradient_descent(self, X, y, y_pred):
        error = y_pred - y
        gradient = np.mean(X.T * error, axis=1)
        regularization_term = self.alpha * np.concatenate([[0], self.coef_[1:]])
        self.coef_ = self.coef_ - self.lr * (gradient + regularization_term)

### test ####
X = np.random.random((100, 3))
Y = np.random.randint(low=0, high=1, size=(100,))

X_val = np.random.random((100, 3))
Y_val = np.random.randint(low=0, high=1, size=(100,))

n_iter = 100
lr = 0.001
alpha = 0.1
bias = True
verbose = True

linear_model = ScratchLogisticRegression(num_iter=n_iter,
                                         lr=lr,
                                         alpha=alpha,
                                         no_bias=bias,
                                         verbose=True)

model, train_loss, val_loss = linear_model.fit(X, Y, X_val, Y_val)

plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.show()
