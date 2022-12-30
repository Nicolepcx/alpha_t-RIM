from scipy.stats import t
import numpy as np
import pandas as pd
from customdslib import preprocessing as pre
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import statsmodels.api as sm
import statsmodels.stats.api as sms


def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : int
        Variance-corrected standard deviation of the set of differences.
    """
    n = n_train + n_test
    corrected_var = (
        np.var(differences, ddof=1) * ((1 / n) + (n_test / n_train))
    )
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples, 1)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def get_rescaled_mape_r2(y_test, predicted, mu, sigma, n_steps_ahead):

    """
    Rescales the data and performs a MAPE and R^2 metrics of given data-

    :param y_test: testing data
    :param predicted: predicted data
    :param mu: mu of the transformed data
    :param sigma: sigma of the transformed data
    :param n_steps_ahead: prediction horizon
    :return: rescaled metrics for the data
    """

    mape_list = []
    r2_list = []
    n_steps_ahead_list = []
    for i in range(n_steps_ahead):
        n_steps_ahead_list.append(i+1)
        test = pre.reverse_scaling(y_test[:, i], mu[0], sigma[0])
        pred = pre.reverse_scaling(predicted[:, i], mu[0], sigma[0])
        mape_list.append((mean_absolute_percentage_error(test, pred)*100))
        r2_list.append(r2_score(test, pred))

    # create dataframe
    metrics_df = pd.DataFrame(list(zip(n_steps_ahead_list, mape_list, r2_list)),
                              columns=['lag', 'MAPE', '$R^2$'])

    # round and remove trailing zeros
    metrics_df['MAPE'] = metrics_df['MAPE'].round(4).astype(str)
    metrics_df['$R^2$'] = metrics_df['$R^2$'].round(4).astype(str)

    # use the lags as index
    metrics_df = metrics_df.set_index('lag')

    return metrics_df


def white_noise_test(residual, n_steps):
    """

    :param residual: models residuals
    :param n_steps: input lags
    :return: performed white noise test results
    """

    lb, p = sm.stats.diagnostic.acorr_ljungbox(residual, lags=n_steps, boxpierce=False)
    n_steps_list = []

    # get lags for dataframe
    for i in range(n_steps):
        n_steps_list.append(i + 1)

    n_steps_arr = np.array(n_steps_list)

    values = np.array(list(zip(n_steps_arr, lb, p)))

    # create dataframe
    white_noise_df = pd.DataFrame(values, columns=['Lag', 'LB-stats', 'P-value'])

    white_noise_df['Lag'].astype(int)

    # round and remove trailing zeros
    white_noise_df['Lag'] = white_noise_df['Lag'].astype(int).astype(str)
    white_noise_df['LB-stats'] = white_noise_df['LB-stats'].round(4).astype(str)
    white_noise_df['P-value'] = white_noise_df['P-value'].round(4).astype(str)

    # use the lags as index
    white_noise_df = white_noise_df.set_index('Lag')

    return white_noise_df


def get_eval_df(model_performance):

    """
    Transorms the eval dictionary into a pandas data frame.

    :param model_performance: models performance dictionary
    :return: metrics as data frame
    """

    metrics_evaluate = pd.DataFrame.from_dict(model_performance).T

    metrics_evaluate = metrics_evaluate.drop(columns=0)
    metrics_evaluate = metrics_evaluate.rename(columns={1: 'MSE', 2: 'MAE'})

    return metrics_evaluate









