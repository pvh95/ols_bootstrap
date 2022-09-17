import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from ols_bootstrap.auxillary.linreg import LR
from ols_bootstrap.auxillary.bca import BCa
from ols_bootstrap.auxillary.std_error import (
    calc_se_orig,
    HC0_1,
    HC2_5,
    homoscedastic_se,
)
from prettytable import PrettyTable, ALL
from IPython.display import display


class BaseEstimator:
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
        self._se_translation = {
            "nonrobust": "nonrobust",
            "hc0": "HC0",
            "hc1": "HC1",
            "hc2": "HC2",
            "hc3": "HC3",
            "hc4": "HC4",
            "hc4m": "HC4m",
            "hc5": "HC5",
        }

        self._ci_translation = {
            "percentile": "Percentile",
            "bc": "BC",
            "bca": "BCa",
            "studentized": "Studentized",
            "basic": "Basic",
            "empirical": "Empirical",
        }

        # Beginning of the optional input arguments check
        if se_type not in self._se_translation:
            raise Exception("Invalid standard error type.")

        if ci_type not in self._ci_translation:
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
            self._X, self._indep_varname = self._init_asarray_X(X, fit_intercept)

        elif isinstance(X, list) or isinstance(X, tuple):
            X = np.asarray(X)
            self._X, self._indep_varname = self._init_asarray_X(X, fit_intercept)

        else:
            raise Exception(
                "X is neither a type of pd.DataFrame nor pd.Series nor np.ndarray nor list nor tuple."
            )

        self._sample_size = self._X.shape[0]
        self._feature_nums = self._X.shape[1]

        # Beginning of checking the "goodness" of the input X:
        if np.isnan(self._X).any():
            raise Exception("There is a NaN value in X.")

        if self._sample_size <= self._feature_nums:
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
            self._Y = self._init_asarray_Y(Y)

        elif isinstance(Y, list) or isinstance(Y, tuple):
            Y = np.asarray(Y)
            self._Y = self._init_asarray_Y(Y)

        else:
            raise Exception(
                "Y is neither a type of pd.DataFrame nor pd.Series nor np.ndarray nor list nor tuple."
            )

        # Beginning of checking the "goodness" of the input Y:
        if np.isnan(self._Y).any():
            raise Exception("There is a NaN value in Y.")

        if self._sample_size != self._Y.shape[0]:
            raise Exception("The number of observations is not equal in X and Y.")

        # End of checking the "goodness" of the input Y:
        # End of the scrutiny of Y:

        self._decode_varname_to_num = {
            key: val for val, key in enumerate(self._indep_varname)
        }

        self._reps = reps
        self._se_type = se_type
        self._ci = ci
        self._ci_type = ci_type
        self._lwb = (1 - self._ci) / 2
        self._upb = self._ci + self._lwb
        self._scale_resid_bool = False

        self._subset_jack_ratio = subset_jack_ratio
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    @staticmethod
    def _init_asarray_X(X_arr, fit_intercept):
        if fit_intercept:
            X_arr = np.c_[np.ones(X_arr.shape[0]), X_arr]
            indep_varname = ["const"] + [
                "x" + str(varnum) for varnum in np.arange(1, X_arr.shape[1])
            ]

        else:
            # If X only contains one feature, reshape it
            if X_arr.shape == (X_arr.shape[0],):
                X_arr = X_arr.reshape(X_arr.shape[0], 1)

            indep_varname = [
                "x" + str(varnum) for varnum in np.arange(1, X_arr.shape[1] + 1)
            ]

        return X_arr, indep_varname

    @staticmethod
    def _init_asarray_Y(Y_arr):
        if Y_arr.shape == (Y_arr.shape[0],):
            return Y_arr

        elif Y_arr.shape == (Y_arr.shape[0], 1):
            Y_arr = Y_arr.reshape(
                Y_arr.shape[0],
            )

            return Y_arr

        elif Y_arr.shape == (1, Y_arr.shape[1]):
            Y_arr = Y_arr.reshape(
                Y_arr.shape[1],
            )

            return Y_arr

        else:
            raise Exception("The shape of Y is not a 1D-alike array.")

    def _calc_orig_param_resid(self):
        model_linreg = LR(self._Y, self._X)
        model_linreg.fit()

        if model_linreg.rank != self._feature_nums:
            raise Exception(
                "The indpendent variables in the sample are not linearly independent!"
            )

        self._orig_params = model_linreg.params
        self._orig_ssr = model_linreg.ssr
        self._orig_rank = model_linreg.rank

        self._orig_pred_train = model_linreg.predict(self._X)
        self._orig_resid = model_linreg.get_residual(self._orig_pred_train)

        self._pinv_XtX = np.linalg.pinv(self._X.T @ self._X)
        if self._ci_type == "studentized" and self._bootstrap_type != "Pairs Bootstrap":
            if self._se_type in {"constant", "hc0", "hc1"}:
                self._H_diag = None
            else:
                self._H_diag = np.diag(self._X @ self._pinv_XtX @ self._X.T)

    def _bootstrap(self):
        pass

    def ols_fit_sample(self, is_statsmodel=True):
        if is_statsmodel:
            X_temp_df = pd.DataFrame(data=self._X, columns=self._indep_varname)

            if self._se_type in ("hc4", "hc4m", "hc5"):
                # Try to use logger instead in the long run
                print(
                    f'For displaying this OLS statsmodel result, HC3 is going to be used as statsmodels currently does not support {self._se_translation[self._se_type]}. To get {self._se_translation[self._se_type]}, please use "get_all_se()" method right after executing the "ols_fit_sample()" method.',
                    end="\n\n",
                )

                statsmodels_model = sm.OLS(self._Y, X_temp_df)
                statsmodels_result = statsmodels_model.fit(cov_type="HC3")
                print(statsmodels_result.summary())

            else:
                statsmodels_model = sm.OLS(self._Y, X_temp_df)
                statsmodels_result = statsmodels_model.fit(
                    cov_type=self._se_translation[self._se_type]
                )
                print(statsmodels_result.summary())

        self._calc_orig_param_resid()
        self._orig_se, self._scaled_residuals = calc_se_orig(
            self._X,
            self._pinv_XtX,
            self._orig_resid,
            self._orig_ssr,
            self._se_type,
            scale_resid_bool=self._scale_resid_bool,
        )

    def fit(self):
        # Check if ols_fit_sample method was run before. If not run it, otherwise don't rerun as it has already been attached to the object.
        if not hasattr(self, "_orig_ssr"):
            self.ols_fit_sample(is_statsmodel=False)

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
        if self._ci_type == "studentized":
            t_stat_pctile = np.percentile(
                self._t_stat_boot, [self._lwb * 100, self._upb * 100], axis=1
            )

            self._ci_mtx = self._orig_params - self._orig_se * t_stat_pctile
            self._ci_mtx[[0, 1]] = self._ci_mtx[
                [1, 0]
            ]  # swap the row because before swapping the first row is the upper-bound

            self._ci_mtx = self._ci_mtx.T

        else:
            bca = BCa(
                self._Y,
                self._X,
                self._orig_params,
                self._indep_vars_bs_param,
                ci=self._ci,
                ci_type=self._ci_type,
                subset_jack_ratio=self._subset_jack_ratio,
                seed=self._seed,
            )

            self._ci_mtx = bca.bca_ci

        self._summary_table()

    def _summary_table(self):
        result_columns = (
            "var",
            "ols_params",
            "avg_bs_params",
            "bias",
            "ols_params_se",
            "bs_params_se",
            "perc_of_se_diff",
            "ci_lwb",
            "ci_upb",
        )

        result_table = np.empty((len(self._indep_varname), len(result_columns))).astype(
            "<U32"
        )

        for idx, var in enumerate(self._indep_varname):
            row_vals = np.array(
                [
                    var,
                    np.round(self._orig_params[idx], 4),
                    np.round(self._indep_vars_bs_mean[idx], 4),
                    np.round(self._indep_vars_bs_bias[idx], 4),
                    np.round(self._orig_se[idx], 4),
                    np.round(self._indep_vars_bs_se[idx], 4),
                    np.round(
                        (1 - self._indep_vars_bs_se[idx] / self._orig_se[idx]) * 100, 2
                    ),
                    np.round(self._ci_mtx[idx, 0], 4),
                    np.round(self._ci_mtx[idx, 1], 4),
                ]
            )
            result_table[idx, :] = row_vals

        self._df_summary = pd.DataFrame(
            data=result_table, columns=result_columns
        ).set_index("var")

    def summary(self):
        display(
            self._df_summary.style.set_caption(
                f"{self._bootstrap_type} results with {self._sample_size} obs and {self._reps} BS reps using {self._se_translation[self._se_type]} SE-s and {(self._ci * 100):.2f}% {self._ci_translation[self._ci_type]} CI"
            )
        )

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
                        f"'{which_var}' does not exist in {self._indep_varname}."
                    )

        else:
            which_var = tuple(which_var)
            row_idx = [self._decode_varname_to_num[key] for key in which_var]
            selected_bs_params = self._indep_vars_bs_param[row_idx].T

        selected_bs_params = pd.DataFrame(data=selected_bs_params, columns=which_var)

        return selected_bs_params

    def get_bca_ci(self, which_ci="current", which_var="all"):
        # all_ci = sorted(["bc", "bca", "percentile"])
        all_ci = sorted(self._ci_translation)
        all_ci.remove("studentized")

        if isinstance(which_ci, str):
            if which_ci == "current":
                if self._ci_type == "studentized":
                    raise Exception(
                        f"{self._ci_type} is not a BCa-type bootstrap confidence interval."
                    )
                else:
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
                # bca = BCa(
                #     self._Y,
                #     self._X,
                #     self._orig_params,
                #     self._indep_vars_bs_param,
                #     ci=self._ci,
                #     ci_type=key,
                # )

                bca = BCa(
                    self._Y,
                    self._X,
                    self._orig_params,
                    self._indep_vars_bs_param,
                    ci=self._ci,
                    ci_type=key,
                    subset_jack_ratio=self._subset_jack_ratio,
                    seed=self._seed,
                )

                selected_ci_dict[key] = bca.bca_ci

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
                        f"'{which_var}' does not exist in {self._indep_varname}."
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

        df_se_mtx = pd.DataFrame(
            data=se_mtx, columns=self._se_translation, index=which_var
        )

        return df_se_mtx

    def bp_test(self, robust=True):
        bp_test_result = het_breuschpagan(
            self._scaled_residuals, self._X, robust=robust
        )

        return bp_test_result

    def white_test(self):
        white_test_result = het_white(self._scaled_residuals, self._X)

        return white_test_result

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

    @property
    def summary_table(self):
        return self._df_summary
