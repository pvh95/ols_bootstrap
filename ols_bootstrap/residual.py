import numpy as np
import pandas as pd

from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap
from prettytable import PrettyTable, ALL


class ResidualBootstrap(PairsBootstrap):
    def __init__(self, Y, X, iter=10000, ci=0.95, is_constant=True):
        super().__init__(Y, X, iter, ci, is_constant)
        self._bootstrap_type = "Residual Bootstrap"

    def _bootstrap(self):
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._iter))

        for i in range(self._iter):
            Y_boot = np.zeros(self._sample_size)
            boot_residuals = np.random.choice(
                self._orig_resid, self._sample_size, replace=True
            )

            for idx in range(self._sample_size):
                Y_boot[idx] = self._orig_pred_train[idx] + boot_residuals[idx]

            ols_model = LR(Y_boot, self._X)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params
