# Homoskedastic, HC0, HC1, HC2 and HC3 attributes' SE-s were tested with statsmodel's appropriate attributes

import numpy as np

### Heteroskedastic Standard Error Calculation Class
class HC0_1:
    def __init__(self, X, residual):
        self._X = X
        self._residual = residual

        pinv_XtX = np.linalg.pinv(self._X.T @ self._X)
        self._cov_HC0_HC1 = (
            pinv_XtX @ self._X.T @ np.diag(self._residual ** 2) @ self._X @ pinv_XtX
        )

    @property
    def HC0_se(self):
        hc0_se = np.sqrt(np.diag(self._cov_HC0_HC1))

        return hc0_se

    @property
    def HC1_se(self):
        self._dof_coeff = self._X.shape[0] / (self._X.shape[0] - self._X.shape[1])
        hc1_se = np.sqrt(self._dof_coeff * np.diag(self._cov_HC0_HC1))

        return hc1_se

    @property
    def HC1_scaled_resid(self):
        hc1_resid = np.sqrt(self._dof_coeff) * self._residual

        return hc1_resid


class HC2_5:
    def __init__(self, X, residual):
        self._X = X
        self._residual = residual

        self._pinv_XtX = np.linalg.pinv(self._X.T @ self._X)
        self._H_diag = np.diag(self._X @ self._pinv_XtX @ self._X.T)

    def _HCCM(self, diag_mtx):
        hccm = self._pinv_XtX @ self._X.T @ diag_mtx @ self._X @ self._pinv_XtX

        return hccm

    ##########################

    @property
    def HC2_se(self):
        self._hc2_resid = self._residual / np.sqrt(1 - self._H_diag)
        cov_HC2 = self._HCCM(np.diag(self._hc2_resid ** 2))

        hc2_se = np.sqrt(np.diag(cov_HC2))

        return hc2_se

    @property
    def HC2_scaled_resid(self):
        return self._hc2_resid

    ##########################

    @property
    def HC3_se(self):
        self._hc3_resid = self._residual / (1 - self._H_diag)
        cov_HC3 = self._HCCM(np.diag(self._hc3_resid ** 2))

        hc3_se = np.sqrt(np.diag(cov_HC3))

        return hc3_se

    @property
    def HC3_scaled_resid(self):
        return self._hc3_resid

    ######################

    @property
    def HC4_se(self):
        delta = np.minimum(4, self._X.shape[0] / self._X.shape[1] * self._H_diag)

        self._hc4_resid = self._residual / np.sqrt((1 - self._H_diag) ** delta)
        cov_HC4 = self._HCCM(np.diag(self._hc4_resid ** 2))

        hc4_se = np.sqrt(np.diag(cov_HC4))

        return hc4_se

    @property
    def HC4_scaled_resid(self):
        return self._hc4_resid

    ######################

    @property
    def HC4m_se(self):
        # https://quantoid.net/files/702/lecture8.pdf -- HC4m Standard Errors

        delta = np.minimum(
            1, self._X.shape[0] / self._X.shape[1] * self._H_diag
        ) + np.minimum(1.5, self._X.shape[0] / self._X.shape[1] * self._H_diag)

        self._hc4m_resid = self._residual / np.sqrt((1 - self._H_diag) ** delta)
        cov_HC4m = self._HCCM(np.diag(self._hc4m_resid ** 2))

        hc4m_se = np.sqrt(np.diag(cov_HC4m))

        return hc4m_se

    @property
    def HC4m_scaled_resid(self):
        return self._hc4m_resid

    ######################

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

        self._hc5_resid = self._residual / np.sqrt((1 - self._H_diag) ** delta)
        cov_HC5 = self._HCCM(np.diag(self._hc5_resid ** 2))

        hc5_se = np.sqrt(np.diag(cov_HC5))

        return hc5_se

    @property
    def HC5_scaled_resid(self):
        return self._hc5_resid


### Homoscedastic Standard Error Calculation Function
def homoscedastic_se(X, ssr):
    const_varience = ssr / (X.shape[0] - X.shape[1])

    cov_mtx_params = const_varience * np.linalg.pinv(X.T @ X)
    homo_se = np.sqrt(np.diag(cov_mtx_params))

    return homo_se


### Calculatign the appropriate standard error
def se_calculation(
    X,
    se_type,
    residual,
    ssr,
    scale_resid=True,
):
    hc_resid = residual

    if se_type == "constant":
        se = homoscedastic_se(X, ssr)

    elif se_type == "hc0":
        hce_basic = HC0_1(X, residual)
        se = hce_basic.HC0_se

    elif se_type == "hc1":
        hce_basic = HC0_1(X, residual)
        se = hce_basic.HC1_se

        if scale_resid:
            hc_resid = hce_basic.HC1_scaled_resid

    elif se_type == "hc2":
        hce_weighted = HC2_5(X, residual)
        se = hce_weighted.HC2_se

        if scale_resid:
            hc_resid = hce_weighted.HC2_scaled_resid

    elif se_type == "hc3":
        hce_weighted = HC2_5(X, residual)
        se = hce_weighted.HC3_se

        if scale_resid:
            hc_resid = hce_weighted.HC3_scaled_resid

    elif se_type == "hc4":
        hce_weighted = HC2_5(X, residual)
        se = hce_weighted.HC4_se

        if scale_resid:
            hc_resid = hce_weighted.HC4_scaled_resid

    elif se_type == "hc4m":
        hce_weighted = HC2_5(X, residual)
        se = hce_weighted.HC4m_se

        if scale_resid:
            hc_resid = hce_weighted.HC4m_scaled_resid

    elif se_type == "hc5":
        hce_weighted = HC2_5(X, residual)
        se = hce_weighted.HC5_se

        if scale_resid:
            hc_resid = hce_weighted.HC5_scaled_resid

    return se, hc_resid
