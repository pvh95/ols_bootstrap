# params() method's result is checked with statsmodel and sklearn's result
# resid(), se(), pred_train() methods' results are checked with statsmodel

import numpy as np


class LR:
    def __init__(self, Y, X):

        if Y.shape == (Y.shape[0],):
            self.__Y = Y.reshape(Y.shape[0], 1)
        else:
            self.__Y = Y

        self.__X = X

    def fit(self):
        self._beta = np.linalg.lstsq(self.__X, self.__Y, rcond=None)[0]
        self._Y_hat_train = self.__X.dot(self._beta)
        self._residual = self.__Y - self._Y_hat_train

        residual_sse = self._residual.T @ self._residual
        sigma_squared_hat = residual_sse[0, 0] / (self.__X.shape[0] - self.__X.shape[1])
        var_beta_hat = np.linalg.inv(self.__X.T @ self.__X) * sigma_squared_hat

        self._beta_se = np.zeros(var_beta_hat.shape[0])

        for idx in range(var_beta_hat.shape[0]):
            self._beta_se[idx] = np.sqrt(var_beta_hat[idx, idx])

    def predict(self, X_test):
        Y_hat_test = X_test.dot(self._beta)

        return Y_hat_test.reshape((Y_hat_test.shape[0],))

    @property
    def params(self):
        return self._beta.reshape((self._beta.shape[0],))

    @property
    def resid(self):
        return self._residual.reshape((self._residual.shape[0],))

    @property
    def se(self):
        return self._beta_se

    @property
    def pred_train(self):
        return self._Y_hat_train.reshape((self._Y_hat_train.shape[0],))
