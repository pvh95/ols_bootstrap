import numpy as np
import pandas as pd
import copy

from ols_bootstrap.auxillary.linreg import LR
from prettytable import PrettyTable, ALL
from collections import OrderedDict


class PairsBootstrap:
    def __init__(self, Y, X, iter=10000, ci=0.95, is_constant=True):

        if is_constant:
            X_mtx = X.to_numpy()
            self._X = np.hstack((np.ones((X_mtx.shape[0], 1)), X_mtx))
            self._indep_varname = ["const"] + X.columns.to_list()

        else:
            self._X = X.to_numpy()
            self._indep_varname = X.columns.to_list()

        self._iter = iter
        self._ci = ci
        self._Y = Y.to_numpy()
        self._sample_size = self._Y.shape[0]
        self._bootstrap_type = "Pairs Bootstrap"

    def _calc_orig_param_resid_se(self):
        model_lin = LR(self._Y, self._X)
        model_lin.fit()

        self._orig_params = model_lin.params
        self._orig_resid = model_lin.resid
        self._orig_se = model_lin.se
        self._orig_pred_train = model_lin.pred_train

    def _bootstrap(self):

        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._iter))

        data_mtx = np.hstack((self._Y, self._X))
        ss = self._sample_size

        for i in range(self._iter):
            idx = np.random.choice(ss, ss, replace=True)
            resampled_mtx = data_mtx[idx]
            Y_resampled = resampled_mtx[:, 0]
            X_resampled = resampled_mtx[:, 1:]

            ols_model = LR(Y_resampled, X_resampled)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params

    def _pct_ci(self):
        lwb = (100 - 100 * self._ci) / 2
        upb = 100 * self._ci + lwb

        pct_ci_mtx = np.zeros((len(self._indep_varname), 2))

        for row in range(pct_ci_mtx.shape[0]):
            pct_ci_mtx[row] = np.percentile(
                self._indep_vars_bs_param[row, :], [lwb, upb]
            )

        return pct_ci_mtx

    def fit(self):
        self._calc_orig_param_resid_se()
        self._bootstrap()
        self._indep_vars_bs_mean = np.mean(
            self._indep_vars_bs_param, axis=1
        )  # Calculating mean for each bootstraped parameter's distro (row mean)
        self._indep_vars_bs_se = np.std(
            self._indep_vars_bs_param, axis=1, ddof=1
        )  # Calculating std for each bootstraped parameter's distro (row std)
        self._indep_vars_bs_bias = np.abs(
            self._orig_params - self._indep_vars_bs_mean
        )  # Calculating Bias

        self._pct_ci_mtx = (
            self._pct_ci()
        )  # Calculating each parameter (row) a confidence interval

    def summary(self):
        table = PrettyTable()
        table.title = f"{self._bootstrap_type} results with sample size of {self._sample_size} and bootstrap resampling size of {self._iter} using {(self._ci * 100):.2f}% CI"
        table.hrules = ALL

        table.field_names = [
            "Params",
            "Orig Coeffs",
            "Mean of Bootstrapped Coeffs",
            "Bias",
            "Orig Coeff SE",
            "SE of Bootstrapped Coeffs",
            "% of Diff in SE",
            "PCT CI",
            "PCT CI Diff",
        ]

        for idx, var in enumerate(self._indep_varname):
            table.add_row(
                [
                    f"{var}",
                    f"{self._orig_params[idx]:.4f}",
                    f"{self._indep_vars_bs_mean[idx]:.4f}",
                    f"{self._indep_vars_bs_bias[idx]:.4f}",
                    f"{self._orig_se[idx]:.4f}",
                    f"{self._indep_vars_bs_se[idx]:.4f}",
                    f"{(1.0 - self._indep_vars_bs_se[idx] / self._orig_se[idx])*100:.2f}",
                    f"[{self._pct_ci_mtx[idx, 0]:.4f}, {self._pct_ci_mtx[idx, 1]:.4f}]",
                    f"{(self._pct_ci_mtx[idx, 1] - self._pct_ci_mtx[idx, 0]):.4f}",
                ]
            )

        table.padding_width = 2
        table.padding_height = 1
        print(table)

    # TODO: parameterek es statisztikak visszadasa, tablazat formazas
