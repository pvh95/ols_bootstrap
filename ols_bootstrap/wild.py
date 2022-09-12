import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.estimator import BaseEstimator
from ols_bootstrap.auxillary.std_error import calc_se_t


class WildBootstrap(BaseEstimator):
    def __init__(
        self,
        Y,
        X,
        reps=50,
        se_type="hc3",
        ci=0.95,
        ci_type="bc",
        fit_intercept=True,
        subset_jack_ratio=None,
        seed=None,
        scale_resid_bool=True,
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
        super().__init__(
            Y, X, reps, se_type, ci, ci_type, fit_intercept, subset_jack_ratio, seed
        )

        self._scale_resid_bool = scale_resid_bool
        self._bootstrap_type = (
            f'Wild Bootstrap with {" ".join(from_distro.split("_")).title()}'
        )

    def _bootstrap(self):
        if self._from_distro == "rademacher":
            rad_val = np.array([1.0, -1.0])
            rad_prob = np.array([0.5, 0.5])

            rv_from_distro_mtx = self._rng.choice(
                rad_val, (self._reps, self._sample_size), replace=True, p=rad_prob
            )

        elif self._from_distro == "webb4":
            # Webb: Reworking Wild Bootstrap Based Inference for Clustered Errors
            webb4_val = np.array(
                [-np.sqrt(3 / 2), -np.sqrt(1 / 2), np.sqrt(1 / 2), np.sqrt(3 / 2)]
            )
            webb4_prob = 1 / 4 * np.ones(4)

            rv_from_distro_mtx = self._rng.choice(
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

            rv_from_distro_mtx = self._rng.choice(
                webb6_val, (self._reps, self._sample_size), replace=True, p=webb6_prob
            )

        elif self._from_distro == "uniform":
            # uniform between -sqrt(3) and sqrt(3)
            # Mackinnon: WILD CLUSTER BOOTSTRAP CONFIDENCE INTERVALS* (2015)
            rv_from_distro_mtx = self._rng.uniform(
                -np.sqrt(3), np.sqrt(3), (self._reps, self._sample_size)
            )

        elif self._from_distro == "standard_normal":
            rv_from_distro_mtx = self._rng.standard_normal(
                (self._reps, self._sample_size)
            )

        elif self._from_distro == "mammen":
            mammen_val = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
            mammen_prob = np.array(
                [
                    (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                    1 - (np.sqrt(5) + 1) / (2 * np.sqrt(5)),
                ]
            )

            rv_from_distro_mtx = self._rng.choice(
                mammen_val, (self._reps, self._sample_size), replace=True, p=mammen_prob
            )

        elif self._from_distro == "mammen_cont":
            # u and w are two independent standard normal distribution
            # Mackinnon: WILD CLUSTER BOOTSTRAP CONFIDENCE INTERVALS* (2015)
            u = self._rng.standard_normal((self._reps, self._sample_size))
            w = self._rng.standard_normal((self._reps, self._sample_size))
            rv_from_distro_mtx = u / np.sqrt(2) + 1 / 2 * (w ** 2 - 1)

        Y_boot_mtx = self._orig_pred_train + self._scaled_residuals * rv_from_distro_mtx

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
