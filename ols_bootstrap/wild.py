import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.pairs import PairsBootstrap


class WildBootstrap(PairsBootstrap):
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
        from_distro="rademacher",
    ):
        # Beginning of the optional input arguments check
        if from_distro not in (
            "rademacher",
            "webb4",
            "webb6",
            "uniform",
            "standard_normal",
            "mammen",
            "mammen_cont",
        ):
            raise Exception("Invalid input for the distributions.")
        # End of the optional input arguments check

        self._from_distro = from_distro
        super().__init__(Y, X, reps, se_type, ci, ci_type, fit_intercept, seed)
        self._bootstrap_type = (
            f'Wild Bootstrap with {" ".join(from_distro.split("_")).title()}'
        )

    def _bootstrap(self):
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._reps))

        if self._from_distro == "rademacher":
            rad_val = np.array([1.0, -1.0])
            rad_prob = np.array([0.5, 0.5])

            rv_from_distro = self._rng.choice(
                rad_val, (self._reps, self._sample_size), replace=True, p=rad_prob
            )

        elif self._from_distro == "webb4":
            # Webb: Reworking Wild Bootstrap Based Inference for Clustered Errors
            webb4_val = np.array(
                [-np.sqrt(3 / 2), -np.sqrt(1 / 2), np.sqrt(1 / 2), np.sqrt(3 / 2)]
            )
            webb4_prob = 1 / 4 * np.ones(4)

            rv_from_distro = self._rng.choice(
                webb4_val, (self._reps, self._sample_size), replace=True, p=webb4_prob
            )

        elif self._from_distro == "webb6":
            # Webb: Reworking Wild Bootstrap Based Inference for Clustered Errors
            webb6_val = np.array(
                [
                    -np.sqrt(3 / 2),
                    -1,
                    -np.sqrt(1 / 2),
                    np.sqrt(1 / 2),
                    1,
                    np.sqrt(3 / 2),
                ]
            )
            webb6_prob = 1 / 6 * np.ones(6)

            rv_from_distro = self._rng.choice(
                webb6_val, (self._reps, self._sample_size), replace=True, p=webb6_prob
            )

        elif self._from_distro == "uniform":
            # uniform between -sqrt(3) and sqrt(3)
            # Mackinnon: WILD CLUSTER BOOTSTRAP CONFIDENCE INTERVALS* (2015)
            rv_from_distro = self._rng.uniform(
                -np.sqrt(3), np.sqrt(3), (self._reps, self._sample_size)
            )

        elif self._from_distro == "standard_normal":
            rv_from_distro = self._rng.standard_normal((self._reps, self._sample_size))

        elif self._from_distro == "mammen":
            mammen_val = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
            mammen_prob = np.array(
                [
                    (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                    1 - (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                ]
            )

            rv_from_distro = self._rng.choice(
                mammen_val, (self._reps, self._sample_size), replace=True, p=mammen_prob
            )

        elif self._from_distro == "mammen_cont":
            # u and v are two independent standard normal distribution
            # Mackinnon: WILD CLUSTER BOOTSTRAP CONFIDENCE INTERVALS* (2015)
            u = self._rng.standard_normal((self._reps, self._sample_size))
            w = self._rng.standard_normal((self._reps, self._sample_size))
            rv_from_distro = u / np.sqrt(2) + 1 / 2 * (w ** 2 - 1)

        for i in range(self._reps):
            Y_boot = self._orig_pred_train + self._orig_resid * rv_from_distro[i]

            ols_model = LR(Y_boot, self._X)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params
