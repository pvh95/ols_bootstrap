import numpy as np
import pandas as pd
from ols_bootstrap.auxillary.linreg import LR
from prettytable import PrettyTable, ALL


class PairsBootstrap:
    def __init__(self, Y, X, iter=10000, ci=0.95, fit_intercept=True):

        if fit_intercept:
            X_mtx = X.to_numpy()
            self._X = np.hstack((np.ones((X_mtx.shape[0], 1)), X_mtx))
            self._indep_varname = ["const"] + X.columns.to_list()

        else:
            self._X = X.to_numpy()
            self._indep_varname = X.columns.to_list()

        self._decode_varname_to_num = {
            key: val for val, key in enumerate(self._indep_varname)
        }

        self._iter = iter

        self._ci = ci
        self._lwb = (1 - self._ci) / 2
        self._upb = self._ci + self._lwb

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
            idx_arr = np.random.choice(ss, ss, replace=True)
            resampled_mtx = data_mtx[idx_arr]
            Y_resampled = resampled_mtx[:, 0]
            X_resampled = resampled_mtx[:, 1:]

            ols_model = LR(Y_resampled, X_resampled)
            ols_model.fit()

            self._indep_vars_bs_param[:, i] = ols_model.params

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

        self._pct_ci_mtx = np.percentile(
            self._indep_vars_bs_param, [self._lwb * 100, self._upb * 100], axis=1
        ).T  # Calculating each parameter (row) a confidence interval

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
                    f"{(1.0 - self._indep_vars_bs_se[idx] / self._orig_se[idx])*100.0:.2f}",
                    f"[{self._pct_ci_mtx[idx, 0]:.4f}, {self._pct_ci_mtx[idx, 1]:.4f}]",
                    f"{(self._pct_ci_mtx[idx, 1] - self._pct_ci_mtx[idx, 0]):.4f}",
                ]
            )

        table.padding_width = 2
        table.padding_height = 1
        print(table)

    def get_bootstap_params(self, which_var="all"):
        if which_var == "all":
            selected_bs_params = self._indep_vars_bs_param.T
            which_var = self._indep_varname

        elif isinstance(which_var, tuple) or isinstance(which_var, list):
            row_idx = [self._decode_varname_to_num[key] for key in which_var]
            selected_bs_params = self._indep_vars_bs_param[row_idx].T

        selected_bs_params = pd.DataFrame(data=selected_bs_params, columns=which_var)

        return selected_bs_params

    def get_pct_ci(self, which_var="all"):
        if which_var == "all":
            selected_ci_params = self._pct_ci_mtx
            which_var = self._indep_varname

        elif isinstance(which_var, tuple) or isinstance(which_var, list):
            row_idx = [self._decode_varname_to_num[key] for key in which_var]
            selected_ci_params = self._pct_ci_mtx[row_idx]

        selected_ci_params = pd.DataFrame(
            data=selected_ci_params, columns=[f"{self._lwb:.3f}", f"{self._upb:.3f}"]
        )
        selected_ci_params.insert(0, "params", which_var)

        return selected_ci_params

    @property
    def indep_varname(self):
        return self._indep_varname

    @property
    def orig_params(self):
        return self._orig_params

    @property
    def bs_params_mean(self):
        return self._indep_vars_bs_mean

    @property
    def orig_params_se(self):
        return self._orig_se

    @property
    def bs_params_se(self):
        return self._indep_vars_bs_se

    # TODO: return parameters and stats, formatting summary table or writing alternative summary() method
