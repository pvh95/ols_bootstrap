import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap


class WildBootstrap(PairsBootstrap):
    def __init__(
        self,
        Y,
        X,
        reps=50,
        ci=0.95,
        ci_type="bc",
        fit_intercept=True,
        from_distro="rademacher",
    ):
        self._from_distro = from_distro
        super().__init__(Y, X, reps, ci, ci_type, fit_intercept)
        self._bootstrap_type = (
            f'Wild Bootstrap with {" ".join(from_distro.split("_")).title()}'
        )

    def _bootstrap(self):
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._reps))

        if self._from_distro == "rademacher":
            rad_val = [1.0, -1.0]
            rad_prob = [0.5, 0.5]

            rv_from_distro = np.random.choice(
                rad_val, self._sample_size, replace=True, p=rad_prob
            )

        elif self._from_distro == "standard_normal":
            rv_from_distro = np.random.standard_normal(self._sample_size)

        elif self._from_distro == "mammen":
            mammen_val = [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2]
            mammen_prob = [
                (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                1 - (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
            ]

            rv_from_distro = np.random.choice(
                mammen_val, self._sample_size, replace=True, p=mammen_prob
            )

        for i in range(self._reps):
            Y_boot = np.zeros(self._sample_size)
            boot_residuals = np.random.choice(
                self._orig_resid, self._sample_size, replace=True
            )

            Y_boot = self._orig_pred_train + boot_residuals * rv_from_distro

            ols_model = LR(Y_boot, self._X)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params
