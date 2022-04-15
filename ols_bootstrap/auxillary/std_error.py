import numpy as np

### Heteroskedastic Standard Error Calculation Class
class HC0_1:
    def __init__(self, X, residual):
        self._X = X

        pinv_XtX = np.linalg.pinv(self._X.T @ self._X)
        diag_mtx_HC0 = np.diag(
            residual ** 2
        )  # equivalent to residual * residual, which is an element-wise operation
        self._cov_HC0_HC1 = pinv_XtX @ self._X.T @ diag_mtx_HC0 @ self._X @ pinv_XtX

    @property
    def HC0_se(self):
        hc0_se = np.sqrt(np.diag(self._cov_HC0_HC1))

        return hc0_se

    @property
    def HC1_se(self):
        hc1_se = np.sqrt(
            np.diag(
                self._X.shape[0]
                / (self._X.shape[0] - self._X.shape[1])
                * self._cov_HC0_HC1
            )
        )

        return hc1_se


class HC2_5:
    def __init__(self, X, residual):
        self._X = X
        self._residual_square_arr = (
            residual ** 2
        )  # equivalent to residual * residual, which is an element-wise operation
        self._pinv_XtX = np.linalg.pinv(self._X.T @ self._X)
        self._H_diag = np.diag(self._X @ self._pinv_XtX @ self._X.T)

    def _HCCM(self, diag_mtx):
        hccm = self._pinv_XtX @ self._X.T @ diag_mtx @ self._X @ self._pinv_XtX

        return hccm

    @property
    def HC2_se(self):
        diag_HC2 = np.diag(self._residual_square_arr / (1 - self._H_diag))
        cov_HC2 = self._HCCM(diag_HC2)

        hc2_se = np.sqrt(np.diag(cov_HC2))

        return hc2_se

    @property
    def HC3_se(self):
        diag_HC3 = np.diag(self._residual_square_arr / (1 - self._H_diag) ** 2)
        cov_HC3 = self._HCCM(diag_HC3)

        hc3_se = np.sqrt(np.diag(cov_HC3))

        return hc3_se

    @property
    def HC4_se(self):
        delta = np.minimum(4, self._X.shape[0] / self._X.shape[1] * self._H_diag)

        diag_HC4 = np.diag(self._residual_square_arr / (1 - self._H_diag) ** delta)
        cov_HC4 = self._HCCM(diag_HC4)

        hc4_se = np.sqrt(np.diag(cov_HC4))

        return hc4_se

    @property
    def HC4m_se(self):
        # https://quantoid.net/files/702/lecture8.pdf -- HC4m Standard Errors

        delta = np.minimum(
            1, self._X.shape[0] / self._X.shape[1] * self._H_diag
        ) + np.minimum(1.5, self._X.shape[0] / self._X.shape[1] * self._H_diag)

        diag_HC4m = np.diag(self._residual_square_arr / (1 - self._H_diag) ** delta)
        cov_HC4m = self._HCCM(diag_HC4m)

        hc4m_se = np.sqrt(np.diag(cov_HC4m))

        return hc4m_se

    @property
    def HC5_se(self):
        # https://www.tandfonline.com/doi/abs/10.1080/03610920601126589?src=recsys&journalCode=lsta20 and
        # https://quantoid.net/files/702/lecture8.pdf -- HC5

        aux_max_hc5 = np.maximum(
            4, self._X.shape[0] / self._X.shape[1] * 0.7 * np.max(self._H_diag)
        )

        delta = np.minimum(
            aux_max_hc5, self._X.shape[0] / self._X.shape[1] * self._H_diag
        )

        diag_HC5 = np.diag(self._residual_square_arr / (1 - self._H_diag) ** delta)
        cov_HC5 = self._HCCM(diag_HC5)

        hc5_se = np.sqrt(np.diag(cov_HC5))

        return hc5_se


### Homoscedastic Standard Error Calculation Function
def homoscedastic_se(X, ssr):
    const_varience = ssr / (X.shape[0] - X.shape[1])

    cov_mtx_params = const_varience * np.linalg.pinv(X.T @ X)
    bse = np.sqrt(np.diag(cov_mtx_params))

    return bse
