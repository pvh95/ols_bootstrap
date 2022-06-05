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
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._reps))

        boot_residuals = self._rng.choice(
            self._orig_resid, (self._reps, self._sample_size), replace=True
        )

        for i in range(self._reps):
            Y_boot = self._orig_pred_train + boot_residuals[i]

            ols_model = LR(Y_boot, self._X)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params
