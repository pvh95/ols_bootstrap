# params() method's result is tested with statsmodel and sklearn's result
# resid(), se(), pred_train() methods' results are tested with statsmodel
# TODO: Needs to rewrite and reorganize the HCE-s section using possibly Sandwich estimators

import numpy as np


class LR_with_HCE:  # Needs to reorganize
    def __init__(self, Y, X):

        if Y.shape == (Y.shape[0], 1):
            self.__Y = Y.reshape(
                Y.shape[0],
            )
        else:
            self.__Y = Y

        self.__X = X
        self.__pinv_XtX = np.linalg.pinv(X.T @ X)

    def _calc_bse(self):
        # Calculate the (constant) standard errors of the parameter estimates
        sigma_squared_hat = self._residual_ssr / (
            self.__X.shape[0] - self.__X.shape[1]
        )  # If not using intercept it should be (self.__X.shape[0] - self.__X.shape[1] - 1) in the denominator

        try:
            # Compute the (Moore-Penrose) pseudo-inverse of (X.T @ X) matrix
            var_beta_hat = sigma_squared_hat * self.__pinv_XtX
            self._bse = np.sqrt(np.diag(var_beta_hat))

        # If the design matrix X is a squared matrix, np.linalg.lstsq() would return an empty array for self._residual_ssr so var_beta_hat will throw an exception
        except ValueError:
            self._bse = np.ones(self.__Y.shape[0]) * np.inf

    def _calc_hce(self):
        # Calculate Heteroskedasticity-Consistent Standard Errors aka White correction
        residual_square_arr = self._residual ** 2
        cov_hc0 = np.diag(
            residual_square_arr
        )  # elementwise multiplications, equivalent: self._residual * self._residual

        high_lev_weight_arr = np.diag(
            self.__X @ self.__pinv_XtX @ self.__X.T
        )  # Taking out the diagonal part of the mtx and substract elementwise from 1
        cov_hc2 = np.diag(residual_square_arr / (1 - high_lev_weight_arr))
        cov_hc3 = np.diag(residual_square_arr / (1 - high_lev_weight_arr) ** 2)

        delta_power_arr_hc4 = np.minimum(
            4, self.__X.shape[0] / self.__X.shape[1] * high_lev_weight_arr
        )
        cov_hc4 = np.diag(
            residual_square_arr / (1 - high_lev_weight_arr) ** delta_power_arr_hc4
        )

        delta_power_arr_hc4_M = np.minimum(
            1, self.__X.shape[0] / self.__X.shape[1] * high_lev_weight_arr
        ) + np.minimum(1.5, self.__X.shape[0] / self.__X.shape[1] * high_lev_weight_arr)
        cov_hc4_M = np.diag(
            residual_square_arr / (1 - high_lev_weight_arr) ** delta_power_arr_hc4_M
        )

        # Try to check HC5 here: https://www.tandfonline.com/doi/abs/10.1080/03610920601126589?src=recsys&journalCode=lsta20
        aux_max_hc5 = np.maximum(
            4, self.__X.shape[0] / self.__X.shape[1] * 0.7 * np.max(high_lev_weight_arr)
        )
        delta_power_arr_hc5 = np.minimum(
            aux_max_hc5, self.__X.shape[0] / self.__X.shape[1] * high_lev_weight_arr
        )
        cov_hc5 = np.diag(
            residual_square_arr / (1 - high_lev_weight_arr) ** delta_power_arr_hc5
        )

        HC0_estim_mtx = (
            self.__pinv_XtX @ self.__X.T @ cov_hc0 @ self.__X @ self.__pinv_XtX
        )

        # HC0 - HC3: Tested with statsmodel
        self._HC0_se = np.sqrt(np.diag(HC0_estim_mtx))
        self._HC1_se = np.sqrt(
            np.diag(
                self.__X.shape[0]
                / (self.__X.shape[0] - self.__X.shape[1])
                * HC0_estim_mtx
            )
        )  # If not using intercept it should be (self.__X.shape[0] - self.__X.shape[1] - 1) in the denominator

        self._HC2_se = np.sqrt(
            np.diag(self.__pinv_XtX @ self.__X.T @ cov_hc2 @ self.__X @ self.__pinv_XtX)
        )
        self._HC3_se = np.sqrt(
            np.diag(self.__pinv_XtX @ self.__X.T @ cov_hc3 @ self.__X @ self.__pinv_XtX)
        )

        # HC4 and HC4m and HC5: https://quantoid.net/files/702/lecture8.pdf ---> needs to verify it with R or Stata, but at first values for HC4_se and HC4m_se seems okay.
        self._HC4_se = np.sqrt(
            np.diag(self.__pinv_XtX @ self.__X.T @ cov_hc4 @ self.__X @ self.__pinv_XtX)
        )
        self._HC4m_se = np.sqrt(
            np.diag(
                self.__pinv_XtX @ self.__X.T @ cov_hc4_M @ self.__X @ self.__pinv_XtX
            )
        )

        self._HC5_se = np.sqrt(
            np.diag(self.__pinv_XtX @ self.__X.T @ cov_hc5 @ self.__X @ self.__pinv_XtX)
        )

    def fit(self):
        lstsq_result = np.linalg.lstsq(self.__X, self.__Y, rcond=None)
        self._beta, self._residual_ssr = lstsq_result[0], lstsq_result[1]

        self._Y_hat_train = self.__X.dot(self._beta)
        self._residual = self.__Y - self._Y_hat_train

        self._calc_bse()  # Calculate the standard errors of the parameter estimates
        self._calc_hce()  # Calculate the HCE of the parameter estimates

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
        return self._bse

    @property
    def pred_train(self):
        return self._Y_hat_train
