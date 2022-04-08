# params() method's result is tested with statsmodel and sklearn's result
# resid(), se(), pred_train() methods' results are tested with statsmodel

import numpy as np


class LR:
    def __init__(self, Y, X):

        if Y.shape == (Y.shape[0], 1):
            self.__Y = Y.reshape(
                Y.shape[0],
            )
        else:
            self.__Y = Y

        self.__X = X

    def fit(self):
        lstsq_result = np.linalg.lstsq(self.__X, self.__Y, rcond=None)
        self._beta, self._residual_ssr = lstsq_result[0], lstsq_result[1]

        self._Y_hat_train = self.__X.dot(self._beta)
        self._residual = self.__Y - self._Y_hat_train

        sigma_squared_hat = self._residual_ssr / (self.__X.shape[0] - self.__X.shape[1])

        try:
            # Compute the (Moore-Penrose) pseudo-inverse of (X.T @ X) matrix
            var_beta_hat = sigma_squared_hat * np.linalg.pinv(self.__X.T @ self.__X)
            self._beta_se = np.sqrt(np.diag(var_beta_hat))

        # If the design matrix X is a squared matrix, np.linalg.lstsq() would return an empty array for self._residual_ssr so var_beta_hat will throw an exception
        except ValueError:
            self._beta_se = np.ones(self.__Y.shape[0]) * np.inf

    def predict(self, X_test):
        Y_hat_test = X_test.dot(self._beta)

        return Y_hat_test

    @property
    def params(self):
        return self._beta

    @property
    def resid(self):
        return self._residual

    @property
    def bse(self):
        return self._beta_se

    @property
    def pred_train(self):
        return self._Y_hat_train
