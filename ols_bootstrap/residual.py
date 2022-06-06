import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap


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
    ):
        super().__init__(Y, X, reps, se_type, ci, ci_type, fit_intercept, seed)
        self._bootstrap_type = "Residual Bootstrap"

    def _bootstrap(self):
        boot_residuals_mtx = self._rng.choice(
            self._orig_resid, (self._reps, self._sample_size), replace=True
        )

        Y_boot_mtx = self._orig_pred_train + boot_residuals_mtx

        # Need to transpose Y_boot_mtx because with transpotion each column would represent a newly created Y for each bootstrap sample.
        ols_model = LR(Y_boot_mtx.T, self._X)
        ols_model.fit()

        self._indep_vars_bs_param = ols_model.params
