import numpy as np
import pandas as pd
import copy


from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap
from prettytable import PrettyTable, ALL
from collections import OrderedDict


class WildBootstrap(PairsBootstrap):
    def __init__(
        self, Y, X, iter=10000, ci=0.95, is_constant=True, from_distro="rademacher"
    ):
        self._from_distro = from_distro
        super().__init__(Y, X, iter, ci, is_constant)
        self._bootstrap_type = (
            f'Wild Bootstrap with {" ".join(from_distro.split("_")).title()}'
        )

    def _bootstrap(self):
        self._indep_vars_bs_param_dict = self._init_bs_vars()

        if self._from_distro == "rademacher":
            rad_val = [1.0, -1.0]
            rad_prob = [0.5, 0.5]

            rv_from_distro = np.random.choice(
                rad_val, size=(self._sample_size,), replace=True, p=rad_prob
            )

        elif self._from_distro == "standard_normal":
            rv_from_distro = np.random.standard_normal((self._sample_size,))

        elif self._from_distro == "mammen":
            mammen_val = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
            mammen_prob = [
                (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                1 - (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
            ]

            rv_from_distro = np.random.choice(
                mammen_val, size=(self._sample_size,), replace=True, p=mammen_prob
            )

        for i in range(self._iter):
            Y_boot = np.zeros(self._sample_size)
            boot_residuals = np.random.choice(
                self._orig_resid, self._sample_size, replace=True
            )

            for idx in range(self._sample_size):
                Y_boot[idx] = (
                    self._orig_pred_train[idx]
                    + boot_residuals[idx] * rv_from_distro[idx]
                )

            ols_model = LR(Y_boot, self._X)
            ols_model.fit()
            ols_param_values = ols_model.params

            for (key, values) in zip(self._indep_vars_bs_param_dict, ols_param_values):
                self._indep_vars_bs_param_dict[key][i] = values
