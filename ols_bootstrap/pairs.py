import numpy as np
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.estimator import BaseEstimator
from ols_bootstrap.auxillary.std_error import calc_se_psb_t


class PairsBootstrap(BaseEstimator):
    # setting the default # of bootstrap resampling to be 50 as in STATA. For precise results, set a higher reps value.
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
    ):

        super().__init__(
            Y, X, reps, se_type, ci, ci_type, fit_intercept, subset_jack_ratio, seed
        )

        self._scale_resid_bool = False
        self._bootstrap_type = "Pairs Bootstrap"

    def _bootstrap(self):
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._reps))
        data_mtx = np.c_[self._Y, self._X]

        # Create an idx_arr_mtx in which each row represents the indicies we use for pairs bootstrap.
        idx_arr_mtx = self._rng.choice(
            self._sample_size, (self._reps, self._sample_size), replace=True
        )

        if self._ci_type == "studentized":
            bs_se = np.zeros((self._feature_nums, self._reps))

            for i in range(self._reps):
                resampled_mtx = data_mtx[idx_arr_mtx[i]]
                Y_resampled = resampled_mtx[:, 0]
                X_resampled = resampled_mtx[:, 1:]

                ols_bs_model = LR(Y_resampled, X_resampled)
                ols_bs_model.fit()

                self._indep_vars_bs_param[:, i] = ols_bs_model.params

                bs_ssr = ols_bs_model.ssr
                bs_pred_train = ols_bs_model.predict(X_resampled)
                bs_resid = ols_bs_model.get_residual(bs_pred_train)

                bs_se[:, i] = calc_se_psb_t(
                    X_resampled, bs_resid, bs_ssr, self._se_type
                )

            self._t_stat_boot = (
                self._indep_vars_bs_param
                - self._orig_params.reshape(self._feature_nums, 1)
            ) / bs_se

        else:
            for i in range(self._reps):
                resampled_mtx = data_mtx[idx_arr_mtx[i]]
                Y_resampled = resampled_mtx[:, 0]
                X_resampled = resampled_mtx[:, 1:]

                ols_bs_model = LR(Y_resampled, X_resampled)
                ols_bs_model.fit()

                self._indep_vars_bs_param[:, i] = ols_bs_model.params
