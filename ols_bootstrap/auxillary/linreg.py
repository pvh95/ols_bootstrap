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
        self._params, self._residual_ssr, self._mtx_rank, _ = np.linalg.lstsq(
            self._X, self._Y, rcond=None
        )

    def predict(self, X_test):
        Y_hat_test = X_test.dot(self._params)

        return Y_hat_test

    def get_residual(self, Y_hat_train):
        residual = self._Y - Y_hat_train

        return residual

    @property
    def params(self):
        return self._params

    @property
    def ssr(self):
        return self._residual_ssr

    @property
    def rank(self):
        return self._mtx_rank
