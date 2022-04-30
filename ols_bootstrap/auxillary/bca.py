import numpy as np
from scipy.stats import norm
from ols_bootstrap.auxillary.linreg import LR


class BCa:
    """
    Implements percentile, BC, BCa confidence interval within the BCa class.
    Percentile is a special case of BCa with acceleration factor a_hat=0 and z0 = 0
    BC is a special case of BCa with acceleration factor a_hat = 0
    """

    def __init__(self, Y, X, orig_params, bs_params, ci=0.95, ci_type="bc"):
        self._Y = Y
        self._X = X
        self._orig_params = orig_params
        self._bs_params = bs_params
        self._ci = ci
        self._lwb = (1 - self._ci) / 2
        self._upb = self._ci + self._lwb
        self._ci_type = ci_type

    def _compute_z0_jknife_reps_acceleration(self):
        self._z0 = np.zeros_like(self._orig_params)
        self._a_hat = np.zeros_like(self._orig_params)

        # Compute z0, the inverse normal distribution function of median bias
        for row_ind in range(self._z0.shape[0]):
            self._z0[row_ind] = norm.ppf(
                np.sum(self._bs_params[row_ind, :] < self._orig_params[row_ind])
                / self._bs_params.shape[1]
            )

        # Compute acceleration factor a_hat. It will be computed from jacknife estimator of the skewness of the parameter if computing for BCa.
        if self._ci_type == "bca":
            jknife_reps = np.zeros_like(self._X)

            for i in range(jknife_reps.shape[0]):
                jknife_X_sample = np.delete(self._X, i, axis=0)
                jknife_Y_sample = np.delete(self._Y, i, axis=0)

                ols_model = LR(jknife_Y_sample, jknife_X_sample)
                ols_model.fit()

                jknife_reps[i, :] = ols_model.params

            mean_jknife_params = np.mean(jknife_reps, axis=0)
            self._a_hat = (1 / 6) * np.divide(
                np.sum((mean_jknife_params - jknife_reps) ** 3, axis=0),
                (np.sum((mean_jknife_params - jknife_reps) ** 2, axis=0) ** (3 / 2)),
            )

    def get_bca_ci(self):
        if self._ci_type == "percentile":
            bca_ci_mtx = np.percentile(
                self._bs_params, [self._lwb * 100, self._upb * 100], axis=1
            ).T

        else:
            self._compute_z0_jknife_reps_acceleration()
            z_lower = norm.ppf(self._lwb)
            z_upper = norm.ppf(self._upb)

            self._ci_lower = np.zeros_like(self._z0)
            self._ci_upper = np.zeros_like(self._z0)

            numerator_in_ci_lower = self._z0 + z_lower
            numerator_in_ci_upper = self._z0 + z_upper

            bca_ci_mtx = np.zeros((self._z0.shape[0], 2))

            for i in range(self._z0.shape[0]):
                self._ci_lower[i] = norm.cdf(
                    self._z0[i]
                    + numerator_in_ci_lower[i]
                    / (1 - self._a_hat[i] * numerator_in_ci_lower[i])
                )
                self._ci_upper[i] = norm.cdf(
                    self._z0[i]
                    + numerator_in_ci_upper[i]
                    / (1 - self._a_hat[i] * numerator_in_ci_upper[i])
                )
                bca_ci_mtx[i, :] = np.percentile(
                    self._bs_params[i],
                    [self._ci_lower[i] * 100, self._ci_upper[i] * 100],
                    axis=0,
                )

        return bca_ci_mtx
