import numpy as np
from scipy.stats import norm
from ols_bootstrap.auxillary.linreg import LR

# TODO:  Rename the file, rename the class
# Not important TODO: maybe implement expanded percentile, abc, double bootstrap if having spare time


class BCa:
    """
    Implements percentile, BC, BCa confidence interval within the BCa class.
    Percentile is a special case of BCa with acceleration factor a_hat=0 and z0 = 0
    BC is a special case of BCa with acceleration factor a_hat = 0
    """

    def __init__(self, Y, X, orig_params, bs_params, ci=0.95, ci_type="bc"):
        self._Y = Y
        self._X = X
        self._orig_params = orig_params.reshape(orig_params.shape[0], 1)
        self._bs_params = bs_params
        self._ci = ci
        self._lwb = (1 - self._ci) / 2
        self._upb = self._ci + self._lwb
        self._ci_type = ci_type

    def _compute_z0_jknife_reps_acceleration(self):
        self._a_hat = np.zeros(self._orig_params.shape[0])
        num_of_bs = self._bs_params.shape[1]

        # Compute z0, the inverse normal distribution function of median bias
        self._z0 = norm.ppf(
            np.sum(self._bs_params < self._orig_params, axis=1) / num_of_bs
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

    @property
    def bca_ci(self):
        if self._ci_type in {"percentile", "basic"}:
            bca_ci_mtx = np.percentile(
                self._bs_params, [self._lwb * 100, self._upb * 100], axis=1
            ).T

            if self._ci_type == "basic":
                bca_ci_mtx = 2 * self._orig_params - bca_ci_mtx
                # swap columns to have the correct lower bound and upper bound columns (in this order).
                bca_ci_mtx[:, [0, 1]] = bca_ci_mtx[:, [1, 0]]

            return bca_ci_mtx

        elif self._ci_type == "empirical":
            delta = self._bs_params - self._orig_params
            delta_pct = np.percentile(delta, [self._upb * 100, self._lwb * 100], axis=1)

            bca_ci_mtx = self._orig_params - delta_pct.T

            return bca_ci_mtx

        else:
            self._compute_z0_jknife_reps_acceleration()
            z_lower = norm.ppf(self._lwb)
            z_upper = norm.ppf(self._upb)

            numerator_in_ci_lower = self._z0 + z_lower
            numerator_in_ci_upper = self._z0 + z_upper

            bca_lower_ppf = norm.cdf(
                self._z0
                + numerator_in_ci_lower / (1 - self._a_hat * numerator_in_ci_lower)
            )

            bca_upper_ppf = norm.cdf(
                self._z0
                + numerator_in_ci_upper / (1 - self._a_hat * numerator_in_ci_upper)
            )

            bca_lwb_ci_val = np.diag(
                np.percentile(self._bs_params, bca_lower_ppf * 100, axis=1)
            )

            bca_upb_ci_val = np.diag(
                np.percentile(self._bs_params, bca_upper_ppf * 100, axis=1)
            )

            bca_ci_mtx = np.c_[bca_lwb_ci_val, bca_upb_ci_val]

        return bca_ci_mtx


# # A slightly modified old verison
# class BCa:
#     """
#     Implements percentile, BC, BCa confidence interval within the BCa class.
#     Percentile is a special case of BCa with acceleration factor a_hat=0 and z0 = 0
#     BC is a special case of BCa with acceleration factor a_hat = 0
#     """

#     def __init__(self, Y, X, orig_params, bs_params, ci=0.95, ci_type="bc"):
#         self._Y = Y
#         self._X = X
#         self._orig_params = orig_params
#         self._bs_params = bs_params
#         self._ci = ci
#         self._lwb = (1 - self._ci) / 2
#         self._upb = self._ci + self._lwb
#         self._ci_type = ci_type

#     def _compute_z0_jknife_reps_acceleration(self):
#         self._z0 = np.zeros_like(self._orig_params)
#         self._a_hat = np.zeros_like(self._orig_params)

#         # Compute z0, the inverse normal distribution function of median bias
#         for row_ind in range(self._z0.shape[0]):
#             self._z0[row_ind] = norm.ppf(
#                 np.sum(self._bs_params[row_ind, :] < self._orig_params[row_ind])
#                 / self._bs_params.shape[1]
#             )

#         # Compute acceleration factor a_hat. It will be computed from jacknife estimator of the skewness of the parameter if computing for BCa.
#         if self._ci_type == "bca":
#             jknife_reps = np.zeros_like(self._X)

#             for i in range(jknife_reps.shape[0]):
#                 jknife_X_sample = np.delete(self._X, i, axis=0)
#                 jknife_Y_sample = np.delete(self._Y, i, axis=0)

#                 ols_model = LR(jknife_Y_sample, jknife_X_sample)
#                 ols_model.fit()

#                 jknife_reps[i, :] = ols_model.params

#             mean_jknife_params = np.mean(jknife_reps, axis=0)
#             self._a_hat = (1 / 6) * np.divide(
#                 np.sum((mean_jknife_params - jknife_reps) ** 3, axis=0),
#                 (np.sum((mean_jknife_params - jknife_reps) ** 2, axis=0) ** (3 / 2)),
#             )

#     @property
#     def bca_ci(self):
#         if self._ci_type == "percentile":
#             bca_ci_mtx = np.percentile(
#                 self._bs_params, [self._lwb * 100, self._upb * 100], axis=1
#             ).T

#         else:
#             self._compute_z0_jknife_reps_acceleration()
#             z_lower = norm.ppf(self._lwb)
#             z_upper = norm.ppf(self._upb)

#             numerator_in_ci_lower = self._z0 + z_lower
#             numerator_in_ci_upper = self._z0 + z_upper

#             bca_lower_ppf = norm.cdf(
#                 self._z0
#                 + numerator_in_ci_lower / (1 - self._a_hat * numerator_in_ci_lower)
#             )

#             bca_upper_ppf = norm.cdf(
#                 self._z0
#                 + numerator_in_ci_upper / (1 - self._a_hat * numerator_in_ci_upper)
#             )

#             bca_ci_mtx = np.zeros((self._z0.shape[0], 2))

#             for i in range(self._z0.shape[0]):
#                 bca_ci_mtx[i, :] = np.percentile(
#                     self._bs_params[i],
#                     [bca_lower_ppf[i] * 100, bca_upper_ppf[i] * 100],
#                     axis=0,
#                 )

#         return bca_ci_mtx
