import numpy as np
import matplotlib.pyplot as plt

def MSE(y_pred, y):
    """
    Parameters
    ----------
    y_pred : ndarray, shape (n_samples,)
    y : ndarray, shape (n_samples,)

    Returns
    ----------
    mse : numpy.float
    """
    mse = np.mean((y-y_pred)**2)

    return mse

class ScratchLinearRegression():
    """
    Parameters
    ----------
    num_iter : int
    lr : float
    no_bias : bool
    verbose : bool

    Attributes
    ----------
    self.coef_ : ndarray, shape (n_features,)
    self.loss : ndarray, shape (self.iter,)
    self.val_loss : ndarray, shape (self.iter,)
    """

    def __init__(self, num_iter, lr, no_bias, verbose):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples, )

        X_val : ndarray, shape (n_samples, n_features)
        y_val : ndarray, shape (n_samples, )
        """
        if self.no_bias == False:
            X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
            if X_val is not None:
                X_val = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1)

        self.coef_ = np.random.random((X.shape[1], ))

        for epoch in range(self.iter):
            y_pred = self._linear_hypothesis(X)
            self.loss[epoch] = np.mean((y-y_pred)**2)
            self._gradient_descent(X, self.loss[epoch])

            ### validation ###
            if X_val is not None:
                y_pred_val = self._linear_hypothesis(X_val)
                self.val_loss[epoch] = np.mean((y_val - y_pred_val) ** 2)

            if self.verbose:
                if X_val is not None:
                    print(f"Epoch-{epoch}: train loss={self.loss[epoch]}, "
                          f"val loss={self.val_loss[epoch]}")
                else:
                    print(f"Epoch-{epoch}: train loss={self.loss[epoch]}")

        return self.coef_, self.loss, self.val_loss

    def predict(self, X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        Returns
        -------
            ndarray, shape (n_samples, 1)
        """
        y_pred = self._linear_hypothesis(X)

        return y_pred

    def _linear_hypothesis(self, X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        Returns
        -------
          ndarray, shape (n_samples, 1)
        """

        y_pred =  np.dot(X, self.coef_)

        return y_pred

    def _gradient_descent(self, X, error):

        gradient = np.mean(X.T * error, axis=1)
        self.coef_ = self.coef_ - self.lr * gradient

        pass

### test ####
X = np.random.random((100, 3))
Y = np.random.random((100, ))

X_val = np.random.random((100, 3))
Y_val = np.random.random((100, ))

n_iter = 5
lr = 1#0.001
bias = True
verbose=True

linear_model = ScratchLinearRegression(num_iter=n_iter,
                                       lr=lr,
                                       no_bias=bias,
                                       verbose=True)

model, train_loss, val_loss = linear_model.fit(X, Y, X_val, Y_val)

plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.show()
