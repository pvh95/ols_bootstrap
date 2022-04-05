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
            self._varnames = tuple(
                Y.columns.to_list() + ["const"] + X.columns.to_list()
            )
        else:
            self._X = X.to_numpy()
            self._varnames = tuple(Y.columns.to_list() + X.columns.to_list())

        self._iter = iter
        self._ci = ci
        self._Y = Y.to_numpy()
        self._sample_size = self._Y.shape[0]
        self._bootstrap_type = "Pairs Bootstrap"

        self._calc_orig_param_resid_se()
        self._indep_vars_params()
        self._indep_vars_se()
        self._bootstrap()
        self._mean()
        self._stde()
        self._bias()
        self._rmse()
        self._ci_lower_bound()
        self._ci_upper_bound()

    def _calc_orig_param_resid_se(self):
        model_lin = LR(self._Y, self._X)
        model_lin.fit()

        self._orig_params = model_lin.params
        self._orig_resid = model_lin.resid
        self._orig_se = model_lin.se
        self._orig_pred_train = model_lin.pred_train

    def _make_init_dict_for_params(self):
        indep_vars_lst = self._varnames[1:]  ### omitting Y
        indep_vars_dict = OrderedDict()

        for key in indep_vars_lst:
            indep_vars_dict[key] = None

        return indep_vars_dict

    def _indep_vars_params(self):
        self._indep_params = self._make_init_dict_for_params()

        for i, key in enumerate(self._indep_params):
            self._indep_params[key] = self._orig_params[i]

    def _indep_vars_se(self):
        self._indep_se = self._make_init_dict_for_params()

        for i, key in enumerate(self._indep_se):
            self._indep_se[key] = self._orig_se[i]

    def _init_bs_vars(self):
        indep_vars_bs_param_dict = self._make_init_dict_for_params()

        for key in indep_vars_bs_param_dict:
            indep_vars_bs_param_dict[key] = np.zeros(self._iter)

        return indep_vars_bs_param_dict

    def _bootstrap(self):
        self._indep_vars_bs_param_dict = self._init_bs_vars()

        data_mtx = np.hstack((self._Y, self._X))
        ss = self._sample_size

        for i in range(self._iter):
            idx = np.random.choice(ss, ss, replace=True)
            resampled_mtx = data_mtx[idx]
            Y_resampled = resampled_mtx[:, 0]
            X_resampled = resampled_mtx[:, 1:]

            ols_model = LR(Y_resampled, X_resampled)
            ols_model.fit()
            ols_param_values = ols_model.params

            for (key, values) in zip(self._indep_vars_bs_param_dict, ols_param_values):
                self._indep_vars_bs_param_dict[key][i] = values

    def _mean(self):
        self._indep_vars_bs_mean = copy.deepcopy(self._indep_vars_bs_param_dict)

        for key in self._indep_vars_bs_mean:
            self._indep_vars_bs_mean[key] = np.mean(self._indep_vars_bs_mean[key])

    def _stde(self):
        self._indep_vars_bs_se = copy.deepcopy(self._indep_vars_bs_param_dict)

        for key in self._indep_vars_bs_se:
            # Calculating with corrected std error
            self._indep_vars_bs_se[key] = np.std(self._indep_vars_bs_se[key], ddof=1)

    def _bias(self):
        self._indep_vars_bs_bias = copy.deepcopy(self._indep_vars_bs_mean)

        for key in self._indep_vars_bs_bias:
            self._indep_vars_bs_bias[key] = np.abs(
                self._indep_params[key] - self._indep_vars_bs_mean[key]
            )

    def _rmse(self):
        self._indep_vars_bs_rmse = copy.deepcopy(self._indep_vars_bs_param_dict)

        for key in self._indep_vars_bs_rmse:
            self._indep_vars_bs_rmse[key] = np.sqrt(
                np.mean((self._indep_params[key] - self._indep_vars_bs_rmse[key]) ** 2)
            )

    def _ci_lower_bound(self):
        self._indep_vars_bs_lwb = copy.deepcopy(self._indep_vars_bs_param_dict)
        lwb = (100 - 100 * self._ci) / 2

        for key in self._indep_vars_bs_lwb:
            self._indep_vars_bs_lwb[key] = np.percentile(
                self._indep_vars_bs_lwb[key], lwb
            )

    def _ci_upper_bound(self):
        self._indep_vars_bs_upb = copy.deepcopy(self._indep_vars_bs_param_dict)
        upb = 100 * self._ci + (100 - 100 * self._ci) / 2

        for key in self._indep_vars_bs_upb:
            self._indep_vars_bs_upb[key] = np.percentile(
                self._indep_vars_bs_upb[key], upb
            )

    def summary(self):
        table = PrettyTable()
        table.title = f"{self._bootstrap_type} results with sample size of {self._sample_size} and bootstrap resampling size of {self._iter} using {(self._ci * 100):.2f}% CI"
        table.hrules = ALL

        table.field_names = [
            "Params",
            "Original Coeff",
            "Mean of Bootstrapped Coeffs",
            "Orig Coeff SE",
            "SE of Bootstrapped Coeffs",
            "% of Diff in SE",
            "Bias",
            "RMSE",
            "CI",
            "CI Diff",
        ]

        orig_coeff = self._indep_params
        orig_se = self._indep_se
        mean_coeff = self._indep_vars_bs_mean
        se_coeff = self._indep_vars_bs_se
        bias_coeff = self._indep_vars_bs_bias
        rmse_coeff = self._indep_vars_bs_rmse
        ci_lwb_coeff = self._indep_vars_bs_lwb
        ci_upb_coeff = self._indep_vars_bs_upb

        for var in orig_coeff:
            table.add_row(
                [
                    f"{var}",
                    f"{orig_coeff[var]:.4f}",
                    f"{mean_coeff[var]:.4f}",
                    f"{orig_se[var]:.4f}",
                    f"{se_coeff[var]:.4f}",
                    f"{(1 - (se_coeff[var] / orig_se[var]))*100:.2f}",
                    f"{bias_coeff[var]:.4f}",
                    f"{rmse_coeff[var]:.4f}",
                    f"[{ci_lwb_coeff[var]:.4f}, {ci_upb_coeff[var]:.4f}]",
                    f"{(ci_upb_coeff[var] - ci_lwb_coeff[var]):.4f}",
                ]
            )

        table.padding_width = 2
        table.padding_height = 1
        print(table)

    # TODO: fit method (lasd Oliverek implementacioja), parameter visszaddas, tablazat formazas
