import numpy as np
import pandas as pd
import itertools
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.auxillary.bca import BCa
from ols_bootstrap.auxillary.std_error import (
    HC0_1,
    HC2_5,
    homoscedastic_se,
    se_calculation,
)
from prettytable import PrettyTable, ALL


class PairsBootstrap:
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
        seed=None,
    ):
        self._se_translation = {
            "constant": "constant",
            "hc0": "HC0",
            "hc1": "HC1",
            "hc2": "HC2",
            "hc3": "HC3",
            "hc4": "HC4",
            "hc4m": "HC4m",
            "hc5": "HC5",
        }

        # Beginning of the optional input arguments check
        if se_type not in self._se_translation:
            raise Exception("Invalid standard error type.")

        if ci_type not in ("bc", "bca", "percentile"):
            raise Exception("Invalid confidence interval type.")
        # End of the optional input arguments check

        # Beginning of the scrutiny of X:
        if isinstance(X, pd.DataFrame):
            if fit_intercept:
                X_mtx = X.to_numpy()
                self._X = np.c_[np.ones(X_mtx.shape[0]), X_mtx]
                self._indep_varname = ["const"] + X.columns.to_list()

            else:
                self._X = X.to_numpy()
                self._indep_varname = X.columns.to_list()

        elif isinstance(X, pd.Series):
            if fit_intercept:
                X_arr = X.to_numpy()
                self._X = np.c_[np.ones(X_arr.shape[0]), X_arr]
                self._indep_varname = ["const", X.name]

            else:
                self._X = X.to_numpy()
                # If X only contains one feature, reshape it
                if self._X.shape == (self._X.shape[0],):
                    self._X = self._X.reshape(self._X.shape[0], 1)

                self._indep_varname = [X.name]

        elif isinstance(X, np.ndarray):
            if fit_intercept:
                self._X = np.c_[np.ones(X.shape[0]), X]
                self._indep_varname = ["const"] + [
                    "x" + str(varnum) for varnum in np.arange(1, self._X.shape[1])
                ]

            else:
                self._X = X
                # If X only contains one feature, reshape it
                if self._X.shape == (self._X.shape[0],):
                    self._X = self._X.reshape(self._X.shape[0], 1)

                self._indep_varname = [
                    "x" + str(varnum) for varnum in np.arange(1, self._X.shape[1] + 1)
                ]

        else:
            raise Exception(
                "X is neither a type of pd.DataFrame nor pd.Series nor np.ndarray"
            )

        # Beginning of checking the "goodness" of the input X:
        if np.isnan(self._X).any() == True:
            raise Exception("There is a NaN value in X.")

        if self._X.shape[0] <= self._X.shape[1]:
            raise Exception(
                "Number of observations is less than or equal to the number of independent variables."
            )

        # End of checking the "goodness" of the input X:
        # End of the scrutiny of X

        # Beginning of the scrutiny of Y:
        if isinstance(Y, pd.DataFrame):
            # As the dataset comes from pd.DataFrame, the dependent variable's shape is (obs, 1). We reshape it to (obs, ) shape.
            self._Y = Y.to_numpy()
            self._Y = self._Y.reshape(
                self._Y.shape[0],
            )

        elif isinstance(Y, pd.Series):
            self._Y = Y.to_numpy()

        elif isinstance(Y, np.ndarray):
            if Y.shape == (Y.shape[0],):
                self._Y = Y

            elif Y.shape == (Y.shape[0], 1):
                self._Y = Y.reshape(
                    Y.shape[0],
                )

            elif Y.shape == (1, Y.shape[1]):
                self._Y = Y.reshape(
                    Y.shape[1],
                )

            else:
                raise Exception("The shape of Y is not 1D-like array")

        else:
            raise Exception(
                "Y is neither a type of pd.DataFrame nor pd.Series nor np.ndarray"
            )

        # Beginning of checking the "goodness" of the input Y:
        if np.isnan(self._Y).any() == True:
            raise Exception("There is a NaN value in Y.")

        if self._X.shape[0] != self._Y.shape[0]:
            raise Exception("The number of observations is not equal in X and Y.")

        # End of checking the "goodness" of the input Y:
        # End of the scrutiny of Y:

        self._sample_size = self._Y.shape[0]

        self._decode_varname_to_num = {
            key: val for val, key in enumerate(self._indep_varname)
        }

        self._reps = reps
        self._se_type = se_type
        self._ci = ci
        self._ci_type = ci_type
        self._lwb = (1 - self._ci) / 2
        self._upb = self._ci + self._lwb

        self._bootstrap_type = "Pairs Bootstrap"

        self._rng = np.random.default_rng(seed)

    def _calc_orig_param_resid_se(self):
        model_linreg = LR(self._Y, self._X)
        model_linreg.fit()

        if model_linreg.rank != self._X.shape[1]:
            raise Exception(
                "The indpendent variables in the sample are not linearly independent!"
            )

        self._orig_params = model_linreg.params
        self._orig_ssr = model_linreg.ssr
        self._orig_rank = model_linreg.rank

        self._orig_pred_train = model_linreg.predict(self._X)
        self._orig_resid = model_linreg.get_residual(self._orig_pred_train)

        self._orig_se = se_calculation(
            self._X, self._se_type, self._orig_resid, self._orig_ssr
        )

    def _bootstrap(self):
        self._indep_vars_bs_param = np.zeros((len(self._indep_varname), self._reps))
        data_mtx = np.c_[self._Y, self._X]
        ss = self._sample_size

        for i in range(self._reps):
            idx_arr = self._rng.choice(ss, ss, replace=True)
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

        # Calculating each parameter (row) a confidence interval
        bca = BCa(
            self._Y,
            self._X,
            self._orig_params,
            self._indep_vars_bs_param,
            ci=self._ci,
            ci_type=self._ci_type,
        )

        self._ci_mtx = bca.get_bca_ci()

    def summary(self):
        ci_translation = {"percentile": "Percentile", "bc": "BC", "bca": "BCa"}

        table = PrettyTable()
        table.title = f"{self._bootstrap_type} results with {self._sample_size} obs and {self._reps} BS reps using {self._se_translation[self._se_type]} SE-s and {(self._ci * 100):.2f}% {ci_translation[self._ci_type]} CI"
        table.hrules = ALL

        table.field_names = [
            "Var",
            "OLS Params",
            "Avg BS Params",
            "Bias",
            "OLS Params SE",
            "BS Params SE",
            "% of SE Diff",
            "CI",
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
                    f"[{self._ci_mtx[idx, 0]:.4f}, {self._ci_mtx[idx, 1]:.4f}]",
                ]
            )

        table.padding_width = 1
        table.padding_height = 1
        print(table)

    def get_bootstrap_params(self, which_var="all"):
        if isinstance(which_var, str):
            if which_var == "all":
                which_var = self._indep_varname
                selected_bs_params = self._indep_vars_bs_param.T

            else:
                if which_var in self._indep_varname:
                    row_idx = [self._decode_varname_to_num[which_var]]
                    which_var = [which_var]

                    selected_bs_params = self._indep_vars_bs_param[row_idx].T

                else:
                    raise Exception(
                        f"'{which_var}' does not exist in {self._indep_varname}"
                    )

        else:
            which_var = tuple(which_var)
            row_idx = [self._decode_varname_to_num[key] for key in which_var]
            selected_bs_params = self._indep_vars_bs_param[row_idx].T

        selected_bs_params = pd.DataFrame(data=selected_bs_params, columns=which_var)

        return selected_bs_params

    def get_ci(self, which_ci="current", which_var="all"):
        all_ci = sorted(["bc", "bca", "percentile"])

        if isinstance(which_ci, str):
            if which_ci == "current":
                which_ci = [self._ci_type]

            elif which_ci == "all":
                which_ci = all_ci

            else:
                which_ci = [which_ci]

        else:
            which_ci = sorted(which_ci)

        possible_combinations = [
            list(val)
            for n in range(1, len(all_ci) + 1)
            for val in itertools.combinations(all_ci, n)
        ]

        if which_ci not in possible_combinations:
            raise Exception(f"{which_ci} is not in {possible_combinations}.")

        selected_ci_dict = dict()

        for key in which_ci:
            if key == self._ci_type:
                selected_ci_dict[key] = self._ci_mtx

            else:
                bca = BCa(
                    self._Y,
                    self._X,
                    self._orig_params,
                    self._indep_vars_bs_param,
                    ci=self._ci,
                    ci_type=key,
                )

                selected_ci_dict[key] = bca.get_bca_ci()

        if isinstance(which_var, str):
            if which_var == "all":
                which_var = self._indep_varname

            else:
                if which_var in self._indep_varname:
                    row_idx = [self._decode_varname_to_num[which_var]]
                    which_var = [which_var]

                    for key in selected_ci_dict:
                        selected_ci_dict[key] = selected_ci_dict[key][row_idx]

                else:
                    raise Exception(
                        f"'{which_var}' does not exist in {self._indep_varname}"
                    )

        else:
            which_var = tuple(which_var)
            row_idx = [self._decode_varname_to_num[key] for key in which_var]

            for key in selected_ci_dict:
                selected_ci_dict[key] = selected_ci_dict[key][row_idx]

        for key in selected_ci_dict:
            selected_ci_dict[key] = pd.DataFrame(
                data=selected_ci_dict[key], columns=[f"lwb", f"upb"], index=which_var
            )

        selected_ci_df = pd.concat(
            selected_ci_dict.values(), axis=1, keys=selected_ci_dict.keys()
        )

        return selected_ci_df

    def get_all_se(self, which_var="all"):

        if isinstance(which_var, str):
            if which_var == "all":
                which_var = self._indep_varname
                idx_lst = [self._decode_varname_to_num[key] for key in which_var]

            else:
                if which_var in self._indep_varname:
                    idx_lst = [self._decode_varname_to_num[which_var]]
                    which_var = [which_var]

                else:
                    raise Exception(
                        f"'{which_var}' does not exist in {self._indep_varname}"
                    )

        else:
            which_var = tuple(which_var)
            idx_lst = [self._decode_varname_to_num[key] for key in which_var]

        se_mtx = np.zeros((len(idx_lst), len(self._se_translation)))

        hce_basic = HC0_1(self._X, self._orig_resid)
        hce_weighted = HC2_5(self._X, self._orig_resid)

        for col_num, se_col in enumerate(self._se_translation):
            if se_col == "bootstrapped":
                se_mtx[:, col_num] = self._indep_vars_bs_se[idx_lst]

            elif se_col == self._se_type:
                se_mtx[:, col_num] = self._orig_se[idx_lst]

            elif se_col == "constant":
                se_mtx[:, col_num] = homoscedastic_se(self._X, self._orig_ssr)[idx_lst]

            elif se_col in ["hc0", "hc1"]:
                if se_col == "hc0":
                    se_mtx[:, col_num] = hce_basic.HC0_se[idx_lst]

                elif se_col == "hc1":
                    se_mtx[:, col_num] = hce_basic.HC1_se[idx_lst]

            else:
                if se_col == "hc2":
                    se_mtx[:, col_num] = hce_weighted.HC2_se[idx_lst]

                elif se_col == "hc3":
                    se_mtx[:, col_num] = hce_weighted.HC3_se[idx_lst]

                elif se_col == "hc4":
                    se_mtx[:, col_num] = hce_weighted.HC4_se[idx_lst]

                elif se_col == "hc4m":
                    se_mtx[:, col_num] = hce_weighted.HC4m_se[idx_lst]

                elif se_col == "hc5":
                    se_mtx[:, col_num] = hce_weighted.HC5_se[idx_lst]

        se_mtx = pd.DataFrame(
            data=se_mtx, columns=self._se_translation, index=which_var
        )

        return se_mtx

    def bp_test(self, robust=True):
        bp_test_result = het_breuschpagan(self._orig_resid, self._X, robust=robust)

        return np.array(bp_test_result)

    def white_test(self):
        white_test_result = het_white(self._orig_resid, self._X)

        return np.array(white_test_result)

    @property
    def indep_varname(self):
        return self._indep_varname

    @property
    def bs_params(self):
        return self._indep_vars_bs_param

    @property
    def orig_params(self):
        return self._orig_params

    @property
    def bs_mean(self):
        return self._indep_vars_bs_mean

    @property
    def orig_se(self):
        return self._orig_se

    @property
    def bs_se(self):
        return self._indep_vars_bs_se

    @property
    def bs_ci(self):
        return self._ci_mtx

    # TODO: formatting summary table or writing alternative summary() method
