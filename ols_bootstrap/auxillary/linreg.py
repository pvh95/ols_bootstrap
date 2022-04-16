# params() method's result was tested with statsmodel and sklearn's results
# resid, ssr, pred_train attributes' results were tested with statsmodel's attributes' results

import numpy as np


class LR:
    def __init__(self, Y, X):

        if Y.shape == (Y.shape[0], 1):
            self._Y = Y.reshape(
                Y.shape[0],
            )
        else:
            self._Y = Y

        self._X = X

    def fit(self):
        self._params, self._residual_ssr, _, _ = np.linalg.lstsq(
            self._X, self._Y, rcond=None
        )

        self._Y_hat_train = self._X.dot(self._params)
        self._residual = self._Y - self._Y_hat_train

    def predict(self, X_test):
        Y_hat_test = X_test.dot(self._params)

        return Y_hat_test

    @property
    def params(self):
        return self._params

    @property
    def ssr(self):
        return self._residual_ssr

    @property
    def resid(self):
        return self._residual

    @property
    def pred_train(self):
        return self._Y_hat_train
