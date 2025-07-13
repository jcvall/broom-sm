
import pandas as pd
import numpy as np
import pandas_flavor as pf
import statsmodels.api as sm
import statsmodels.stats as stp
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
import bayesian_bootstrap as bb
from scipy import stats 
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import sys


__author__ = "John C Vallier"
__copyright__ = "John C Vallier"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

# --- Model Configuration ---
MODEL_CONFIG = {
    "ols": {"fitter": smf.ols, "stat_name": "t_stat", "has_rsq": True},
    "logit": {"fitter": smf.logit, "stat_name": "z_stat", "has_rsq": False, "pseudo_rsq_attr": "prsquared"},
    "poisson": {"fitter": lambda f, d, **k: smf.glm(f, d, family=sm.families.Poisson(), **k), "stat_name": "z_stat", "has_rsq": False},
    "gamma": {"fitter": lambda f, d, **k: smf.glm(f, d, family=sm.families.Gamma(), **k), "stat_name": "z_stat", "has_rsq": False},
    "glm": {"fitter": smf.glm, "stat_name": "z_stat", "has_rsq": False}, # For generic GLM, R-squared might not be standard
    "quantreg": {"fitter": smf.quantreg, "stat_name": "t_stat", "has_rsq": False} # Has pseudo R-squared often
}

@pf.register_dataframe_method
def stats_tidy(data: pd.DataFrame, formula: str, stat_type: str, **kwargs):
    """
    Generates a tidy summary of model coefficients and statistics.
    """
    if stat_type not in MODEL_CONFIG:
        raise ValueError(f"Unsupported stat_type: {stat_type}. Supported types are: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[stat_type]
    fitter = config["fitter"]
    stat_name = config["stat_name"]

    model = fitter(formula, data=data, **kwargs).fit()

    params = model.params.to_frame().reset_index().rename(columns={"index": "term", 0: "estimate"})
    conf_int = model.conf_int().reset_index().rename(columns={"index": "term", 0: "conf.low", 1: "conf.high"})
    bse = model.bse.to_frame().reset_index().rename(columns={"index": "term", 0: "std.error"})
    
    if hasattr(model, 'tvalues'):
        stat_values = model.tvalues.to_frame().reset_index().rename(columns={"index": "term", 0: "statistic"})
    elif hasattr(model, 'zvalues'): # some models might use zvalues
         stat_values = model.zvalues.to_frame().reset_index().rename(columns={"index": "term", 0: "statistic"})
    else: # Fallback if neither tvalues nor zvalues, though unlikely for common models
        stat_values = pd.DataFrame(columns=['term', 'statistic'])


    p_values = model.pvalues.to_frame().reset_index().rename(columns={"index": "term", 0: "p.value"})

    df = (
        params
        .merge(conf_int, on="term", how="left")
        .merge(bse, on="term", how="left")
        .merge(stat_values, on="term", how="left")
        .merge(p_values, on="term", how="left")
    )
    
    # Rename statistic column based on model type (t_stat or z_stat)
    df.rename(columns={"statistic": stat_name}, inplace=True)
    
    return df


@pf.register_dataframe_method
def stats_glance(data: pd.DataFrame, formula: str, stat_type: str, **kwargs):
    """
    Generates a one-row summary of model statistics.
    """
    if stat_type not in MODEL_CONFIG:
        raise ValueError(f"Unsupported stat_type: {stat_type}. Supported types are: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[stat_type]
    fitter = config["fitter"]
    model = fitter(formula, data=data, **kwargs).fit()

    glance_dict = {}

    # Common attributes
    if hasattr(model, 'nobs'): glance_dict["n_obs"] = model.nobs
    if hasattr(model, 'llf'): glance_dict["log_likelihood"] = model.llf
    if hasattr(model, 'aic'): glance_dict["aic"] = model.aic
    if hasattr(model, 'bic'): glance_dict["bic"] = model.bic
    if hasattr(model, 'df_model'): glance_dict["df_model"] = model.df_model # df for model terms
    if hasattr(model, 'df_resid'): glance_dict["df_resid"] = model.df_resid
    
    # R-squared and variants
    if config.get("has_rsq", False):
        if hasattr(model, 'rsquared'): glance_dict["rsquared"] = model.rsquared
        if hasattr(model, 'rsquared_adj'): glance_dict["rsquared_adj"] = model.rsquared_adj
    
    pseudo_rsq_attr = config.get("pseudo_rsq_attr")
    if pseudo_rsq_attr and hasattr(model, pseudo_rsq_attr):
        glance_dict["pseudo_rsquared"] = getattr(model, pseudo_rsq_attr)
    elif stat_type == "quantreg" and hasattr(model, 'prsquared'): # QuantReg specific
         glance_dict["pseudo_rsquared"] = model.prsquared

    # OLS specific F-statistic
    if stat_type == "ols":
        if hasattr(model, 'fvalue'): glance_dict["f_statistic"] = model.fvalue
        if hasattr(model, 'f_pvalue'): glance_dict["f_pvalue"] = model.f_pvalue
        if hasattr(model, 'scale'): glance_dict["sigma"] = np.sqrt(model.scale)

    # GLM/Logit specific (deviance, etc.) - can be expanded
    if stat_type in ["glm", "logit", "poisson", "gamma"]:
        if hasattr(model, 'deviance'): glance_dict["deviance"] = model.deviance
        if hasattr(model, 'pearson_chi2'): glance_dict["pearson_chi2"] = model.pearson_chi2
        if hasattr(model, 'scale'): glance_dict["scale"] = model.scale # Dispersion for GLM

    # Ensure all values are scalar or list-like for from_dict
    for key, value in glance_dict.items():
        if isinstance(value, (pd.Series, np.ndarray)) and value.size == 1:
            glance_dict[key] = value.item()
        elif isinstance(value, (list, tuple)) and len(value) == 1:
             glance_dict[key] = value[0]

    df = pd.DataFrame.from_dict(glance_dict, orient="index").transpose()
    return df


@pf.register_dataframe_method
def stats_augment(data: pd.DataFrame, formula: str, stat_type: str, **kwargs):
    """
    Augments the original data with model predictions and diagnostics.
    """
    if stat_type not in MODEL_CONFIG:
        raise ValueError(f"Unsupported stat_type: {stat_type}. Supported types are: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[stat_type]
    fitter = config["fitter"]
    model_fit = fitter(formula, data=data, **kwargs).fit()

    # Prepare original data by getting y and X (excluding intercept)
    # Use the original data's index for proper alignment
    y_df, X_df = dmatrices(formula, data=data, return_type="dataframe")
    
    # Ensure X_df uses the original data's index if patsy changes it
    X_df = X_df.set_index(data.index)
    y_df = y_df.set_index(data.index)

    # Start with original data columns used in the model
    # This ensures we only keep relevant parts of the original data
    # and maintain the original index.
    output_df = data[list(y_df.columns) + [col for col in X_df.columns if col != "Intercept"]].copy()

    output_df[".fitted"] = model_fit.fittedvalues.values
    output_df[".resid"] = model_fit.resid # Or model_fit.resid for some models

    if hasattr(model_fit, 'get_prediction'):
        predictions = model_fit.get_prediction()
        if hasattr(predictions, 'se_mean'):
            output_df[".se.fit"] = predictions.se_mean

    if hasattr(model_fit, 'get_influence'):
        influence = model_fit.get_influence()
        if hasattr(influence, 'hat_matrix_diag'):
            output_df[".hat"] = influence.hat_matrix_diag
        if hasattr(influence, 'cooks_distance') and len(influence.cooks_distance[0]) == len(data): # cooks_distance can be tuple
            output_df[".cooksd"] = influence.cooks_distance[0]
        if hasattr(influence, 'resid_studentized_internal'):
            output_df[".std.resid"] = influence.resid_studentized_internal
        elif hasattr(influence, 'resid_pearson'): # Fallback for some GLMs
            output_df[".std.resid"] = influence.resid_pearson
            
    return output_df


@pf.register_dataframe_method
def boot_tidy(data: pd.DataFrame, formula: str, stat_type: str, n_boot: int, seed: int = None, **kwargs):
    """
    Performs bootstrap resampling and returns tidy summaries for each bootstrap sample.
    """
    results_list = []
    for i in range(n_boot):
        current_seed = seed + i if seed is not None else None
        resampled_data = data.sample(n=len(data), replace=True, random_state=current_seed)
        try:
            tidy_df = resampled_data.stats_tidy(formula, stat_type, **kwargs)
            tidy_df[".bootstrap_id"] = i
            results_list.append(tidy_df)
        except Exception as e:
            print(f"Warning: Bootstrap sample {i} failed for tidy: {e}")
            # Optionally, append a row with NaNs or skip
    if not results_list:
        return pd.DataFrame()
    return pd.concat(results_list).reset_index(drop=True)


@pf.register_dataframe_method
def boot_glance(data: pd.DataFrame, formula: str, stat_type: str, n_boot: int, seed: int = None, **kwargs):
    """
    Performs bootstrap resampling and returns glance summaries for each bootstrap sample.
    """
    results_list = []
    for i in range(n_boot):
        current_seed = seed + i if seed is not None else None
        resampled_data = data.sample(n=len(data), replace=True, random_state=current_seed)
        try:
            glance_df = resampled_data.stats_glance(formula, stat_type, **kwargs)
            glance_df[".bootstrap_id"] = i
            results_list.append(glance_df)
        except Exception as e:
            print(f"Warning: Bootstrap sample {i} failed for glance: {e}")
    if not results_list:
        return pd.DataFrame()
    return pd.concat(results_list).reset_index(drop=True)


@pf.register_dataframe_method
def boot_augment(data: pd.DataFrame, formula: str, stat_type: str, n_boot: int, seed: int = None, **kwargs):
    """
    Performs bootstrap resampling and returns augmented data for each bootstrap sample.
    Note: This can produce a very large DataFrame.
    """
    results_list = []
    for i in range(n_boot):
        current_seed = seed + i if seed is not None else None
        # Augment needs to apply to the *original* data using a model fit on *resampled* data.
        # Or, augment the resampled data itself. The latter is more common for diagnostics on bootstrapped models.
        # Let's assume augmenting the resampled data.
        resampled_data_for_fit = data.sample(n=len(data), replace=True, random_state=current_seed)
        try:
            # Augment function should ideally take the data to augment as an argument
            # For now, we augment the resampled data itself.
            augment_df = resampled_data_for_fit.stats_augment(formula, stat_type, **kwargs)
            augment_df[".bootstrap_id"] = i
            results_list.append(augment_df)
        except Exception as e:
            print(f"Warning: Bootstrap sample {i} failed for augment: {e}")
    if not results_list:
        return pd.DataFrame()
    return pd.concat(results_list).reset_index(drop=True)


def stats_power(effect_size: float, alpha: float, power: float = None, obs_range: tuple = (2, 50), nobs: int = None):
    """
    Performs power analysis for an independent t-test and plots power curves.
    Can solve for sample size if power is provided, or plot power for a range of N if nobs is not fixed.
    """

    power_analysis = stp.power.TTestIndPower()
    
    if power is not None and nobs is None:
        sample_size = power_analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0, alternative='two-sided')
        print(f'Required sample size (per group): {sample_size:.2f}')
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) # Use a single subplot
    ax.set_ylabel("power")
    
    plot_nobs = np.arange(obs_range[0], obs_range[1]) if nobs is None else np.array([nobs])
    
    power_analysis.plot_power(dep_var='nobs',
                                 nobs= plot_nobs,
                                 effect_size=np.array([effect_size]),
                                 ax=ax, title=r'alpha = %1.3f' % alpha) 
    plt.show()


@pf.register_dataframe_method
def stats_residual_plot(data: pd.DataFrame, x: list, y: str):

    for i in x:
        # Residual plot makes sense against predictors of a model, or fitted values.
        # Shapiro-Wilk is for normality of a single variable (e.g., residuals).
        # This function seems to mix concepts. Let's assume `data[i]` are residuals if y is the original outcome.
        # Or, if `data` is an augmented frame, `i` could be a predictor and `y` the residual column.
        # For now, interpreting `data[i]` as the variable to check for normality and plot residuals against.
        
        # If `data[i]` is intended to be residuals:
        if data[i].dtype.kind not in 'biufc': # Check if numeric
            print(f"Skipping non-numeric column for Shapiro-Wilk: {i}")
            continue
        
        shap_stat, shap_p = stats.shapiro(data[i])
        
        print(f'\n--- Diagnostic plots for {i} ---')
        print(f'Shapiro-Wilk Test for {i}: Statistic={shap_stat:.3f}, P-Value={shap_p:.4f}')
        
        plt.subplot(1, 2, 1)
        # If `y` is the name of the residual column and `i` is a predictor:
        # sns.residplot(x=data[i], y=data[y], lowess=True, color='green')
        # If `i` is the predictor and `y` is the outcome (for a model's residuals):
        # This requires a model to be fit first. The current signature is ambiguous.
        # Assuming `y` is the outcome, and we want to see `i` vs residuals of `y ~ i`.
        # This is complex to do generically here.
        # A simpler interpretation: plot `i` vs `y` and a probability plot of `i`.
        sns.scatterplot(x=data[i], y=data[y], color='green', alpha=0.5)
        plt.title(f'Scatter: {i} vs {y}')
        
        plt.subplot(1, 2, 2)
        stats.probplot(data[i], plot=sns.mpl.pyplot)
        plt.title(f'Probplot of {i}')
        
        plt.tight_layout()
        plt.show()


@pf.register_dataframe_method
def stats_ols_plot(data: pd.DataFrame, x: list, y: str):

    for i in x:
        sns.jointplot(x = i, y = y, data=data, kind="reg")
        plt.show()


@pf.register_dataframe_method
def stats_influence_plot(data: pd.DataFrame, formula: str, stat_type: str = "ols", alpha: float = 0.05, **kwargs):
  
    if stat_type not in MODEL_CONFIG:
        raise ValueError(f"Unsupported stat_type: {stat_type}. Supported types are: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[stat_type]
    fitter = config["fitter"]
    model = fitter(formula, data=data, **kwargs).fit()
    
    fig = sm.graphics.influence_plot(model, alpha=alpha, criterion="cooks")
    fig.tight_layout(pad=1.0)
    plt.show()	

@pf.register_dataframe_method	
def stats_chisquare_plot(data: pd.DataFrame, x: str, y: str):
    """
    Performs a Chi-square test of independence and plots a heatmap of observed vs expected.
    """
    # Ensure columns are categorical or string for crosstab
    cross_tab = pd.crosstab(data[y].astype('category'), data[x].astype('category'))
    
    chi2, p, dof, expected = chi2_contingency(cross_tab, correction=True)
    
    print(f"Chi-square Test of Independence ({x} vs {y}):")
    print(f"Chi2 Statistic: {chi2:.3f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    
    plt.figure(figsize=(10, 7))
    # Plotting residuals (observed - expected) might be more informative than expected - observed
    residuals = cross_tab - expected
    sns.heatmap(residuals, annot=True, fmt=".1f", cmap="coolwarm", center=0)
    plt.title(f'Heatmap of Observed - Expected Frequencies\nChi2 p-value: {p:.4f}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
    return {"chi2": chi2, "p_value": p, "dof": dof, "observed": cross_tab, "expected": pd.DataFrame(expected, index=cross_tab.index, columns=cross_tab.columns)}

@pf.register_dataframe_method
def stats_vif(data: pd.DataFrame, formula: str = None):
    """
    Calculates Variance Inflation Factor (VIF) for predictors.
    If formula is provided, uses it to determine predictors. Otherwise, uses all numeric columns.
    Requires an intercept in the design matrix for correct VIF calculation.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if formula:
        _, X_df = dmatrices(formula, data=data, return_type="dataframe")
        # Ensure Intercept is present for VIF calculation if not explicitly removed
        if 'Intercept' not in X_df.columns:
             X_df = sm.add_constant(X_df, has_constant='add')
    else:
        X_df = data.select_dtypes(include=np.number)
        # Check if an intercept-like column (all ones) exists, if not, add one.
        if not any((X_df == 1).all()):
            X_df = sm.add_constant(X_df, prepend=True, has_constant='add')

    if 'Intercept' in X_df.columns:
        # VIF is typically not calculated for the intercept itself if it was added.
        # However, variance_inflation_factor needs the full matrix including intercept.
        # We iterate over columns that are not the intercept.
        predictor_cols = [col for col in X_df.columns if col != 'Intercept']
        if not predictor_cols: # Only intercept
            return pd.Series(dtype=float, name="VIF")
        
        vif_data = {col: variance_inflation_factor(X_df.values, X_df.columns.get_loc(col)) 
                    for col in predictor_cols}
    else: # No intercept column found or explicitly handled
        vif_data = {X_df.columns[i]: variance_inflation_factor(X_df.values, i)
                    for i in range(X_df.shape[1])}

    vif_series = pd.Series(vif_data, name="VIF")
    vif_series = pd.Series(d)
    return vif_series

@pf.register_dataframe_method
def stats_formula(data: pd.DataFrame, target: str, *exclude: str): # Changed exclude to *str for direct string args
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    '''
    df_columns = list(data.columns.values)
    df_columns.remove(target)
    if exclude: # exclude can be a tuple of strings
        for col in exclude:
            if col in df_columns: df_columns.remove(col)
    return target + ' ~ ' + ' + '.join(df_columns)

@pf.register_dataframe_method
def stats_conprob(data: pd.DataFrame, var_A_name: str, var_B_name: str, P_A_given_B: bool = True):
    '''
    conditional probability table a given b or p(A|B) using pd.crosstab.
    This doesn't use the chaining method, nor can you use a categorical variables.
    '''
    if transpose==0:
        table=pd.crosstab(A,B)
    else:
        table=pd.crosstab(B,A)
    cnames=table.columns.values
    weights=1/table[cnames].sum()
    out=table*weights
    pc=table[cnames].sum()/table[cnames].sum().sum()
    table=table.transpose()
    cnames=table.columns.values
    p=table[cnames].sum()/table[cnames].sum().sum()
    out['p']=p
    return out


@pf.register_dataframe_method
def bayes_boot(df: pd.DataFrame, target_column : str, n_samples : int):
    '''bootstrap on target column and reutrns a new column on dataframe using bayesian methods'''
    sample = bb.mean(df[target_column].dropna(axis = 0).values, n_samples)
    return pd.Series(sample)


@pf.register_dataframe_method
def bayes_boot_plot(df: pd.DataFrame, columns: list, x_label: str, title: str):
    
    for i in columns:
        plt.figure() # Create a new figure for each plot
        # Use histplot for distribution and kdeplot for density line if desired
        sns.histplot(df[i], kde=True, label='Bayesian Bootstrap Samples', stat="density")
        l, r = bb.highest_density_interval(df[i])
        plt.plot([l, r], [0, 0], linewidth=5.0, marker='o', label='95% HDI')
        plt.xlabel(x_label)
        plt.ylabel('Density')
        plt.title(f"{title} (Column: {i})")
        plt.legend()
        sns.despine()
        plt.show()

# --- New Standalone Tidy Functions ---

@pf.register_dataframe_method
def stats_anova_tidy(data: pd.DataFrame, formula: str, anova_type: int = 2, **kwargs):
    """
    Performs ANOVA on a fitted OLS model and returns a tidy DataFrame.
    """
    model = smf.ols(formula, data=data, **kwargs).fit()
    anova_table = sm.stats.anova_lm(model, typ=anova_type)
    
    # Clean up the anova_table DataFrame
    anova_table = anova_table.reset_index().rename(columns={'index': 'term'})
    if 'PR(>F)' in anova_table.columns:
        anova_table.rename(columns={'PR(>F)': 'p.value'}, inplace=True)
    if 'F' in anova_table.columns:
        anova_table.rename(columns={'F': 'statistic'}, inplace=True)
    
    return anova_table

@pf.register_dataframe_method
def stats_kruskal_tidy(data: pd.DataFrame, value_col: str, group_col: str):
    """
    Performs the Kruskal-Wallis H-test for independent samples.
    Returns a tidy DataFrame with the test statistic and p-value.
    """
    if data[value_col].isnull().any():
        print(f"Warning: Column '{value_col}' contains NaNs. Kruskal-Wallis may fail or produce unexpected results.")
    if data[group_col].isnull().any():
        print(f"Warning: Column '{group_col}' contains NaNs. Kruskal-Wallis may fail or produce unexpected results.")
        
    samples = [group[value_col].dropna().values for name, group in data.groupby(group_col)]
    
    # Ensure there are at least two groups and they are not empty
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2:
        print("Warning: Kruskal-Wallis test requires at least two non-empty groups.")
        return pd.DataFrame({'statistic': [np.nan], 'p.value': [np.nan], 'df': [np.nan]})
        
    statistic, p_value = stats.kruskal(*samples)
    df = len(samples) - 1 # Degrees of freedom
    
    return pd.DataFrame({'statistic': [statistic], 'p.value': [p_value], 'df': [df]})

@pf.register_dataframe_method
def stats_correlation_tidy(data: pd.DataFrame, col1: str = None, col2: str = None, method: str = 'pearson', columns: list = None):
    """
    Calculates Pearson or Spearman correlation.
    - If col1 and col2 are provided, computes correlation between these two.
    - If columns list is provided, computes pairwise correlations for these columns.
    - If no specific columns are provided, computes for all numeric columns in data.
    Returns a tidy DataFrame.
    """
    if method not in ['pearson', 'spearman']:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    corr_func = stats.pearsonr if method == 'pearson' else stats.spearmanr
    results = []

    if col1 and col2:
        if data[col1].isnull().any() or data[col2].isnull().any():
            print(f"Warning: Columns '{col1}' or '{col2}' contain NaNs. Correlation will be calculated on non-NaN pairs.")
        # Drop NaN pairs for the calculation
        valid_data = data[[col1, col2]].dropna()
        if len(valid_data) < 2:
             print(f"Warning: Not enough non-NaN pairs for correlation between {col1} and {col2}.")
             stat, p_val = np.nan, np.nan
        else:
            stat, p_val = corr_func(valid_data[col1], valid_data[col2])
        results.append({'term1': col1, 'term2': col2, 'correlation': stat, 'p.value': p_val})
    else:
        if columns:
            num_df = data[columns].select_dtypes(include=np.number)
        else:
            num_df = data.select_dtypes(include=np.number)
        
        from itertools import combinations
        for c1, c2 in combinations(num_df.columns, 2):
            valid_data = num_df[[c1, c2]].dropna()
            if len(valid_data) < 2:
                stat, p_val = np.nan, np.nan
            else:
                stat, p_val = corr_func(valid_data[c1], valid_data[c2])
            results.append({'term1': c1, 'term2': c2, 'correlation': stat, 'p.value': p_val})
            
    return pd.DataFrame(results)
