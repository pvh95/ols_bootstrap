import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap
from ols_bootstrap.auxillary.std_error import calc_se_t


class ResidualBootstrap(PairsBootstrap):
    def __init__(
        self,
        Y,
        X,
        reps=50,
        se_type="hc3",
        ci=0.95,
        ci_type="bc",
        fit_intercept=True,
        seed=None,
        scale_resid_bool=True,
    ):

        super().__init__(Y, X, reps, se_type, ci, ci_type, fit_intercept, seed)
        self._scale_resid_bool = scale_resid_bool
        self._bootstrap_type = "Residual Bootstrap"

    def _bootstrap(self):
        boot_residuals_mtx = self._rng.choice(
            self._scaled_residuals, (self._reps, self._sample_size), replace=True
        )

        Y_boot_mtx = self._orig_pred_train + boot_residuals_mtx

        # Need to transpose Y_boot_mtx because with transpotion each column would represent a newly created Y for each bootstrap sample.
        ols_bs_model = LR(Y_boot_mtx.T, self._X)
        ols_bs_model.fit()

        self._indep_vars_bs_param = ols_bs_model.params

        if self._ci_type == "studentized":
            bs_ssr = ols_bs_model.ssr
            bs_pred_train = ols_bs_model.predict(self._X)
            bs_resid = ols_bs_model.get_residual(bs_pred_train)

            bs_se = np.zeros((self._feature_nums, self._reps))

            for i in range(self._reps):
                bs_se[:, i] = calc_se_t(
                    self._X,
                    self._pinv_XtX,
                    bs_resid[:, i],
                    bs_ssr[i],
                    self._se_type,
                    H_diag=self._H_diag,
                )

            self._t_stat_boot = (
                self._indep_vars_bs_param
                - self._orig_params.reshape(self._feature_nums, 1)
            ) / bs_se
