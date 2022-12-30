import os
import matplotlib.pyplot as plt
import pandas as pd

# imports for technical analysis
#import talib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from tsmoothie.smoother import *
from pandas.plotting import lag_plot
import seaborn as sns
import statsmodels as sm
from sklearn.metrics import r2_score, mean_squared_error

import scipy.stats as stats
from customdslib import preprocessing as pre


from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr


# save the images of the plots for the report
images_path = '../images/'


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """
    Saves the images of the generated plots as png with a resolution of 300dpi.

    :param fig_id: file name
    :param tight_layout: True or False
    :param fig_extension: file format
    :param resolution: resolution of the plot
    :return: saved version of the plot
    """

    path = os.path.join(images_path, fig_id + '.' + fig_extension)
    print('Saving figure: ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def r2_error_plot(y_true, y_pred, n_steps_ahead, title):

    """

    :param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: plot with r^2 score for each predicted lag
    """

    r2_lag = []
    plt.grid(True)
    for i in range(5):
        r2_lag.append(r2_score(y_true[:, i], y_pred[:, i]))
        #print(r2_score(y_true[:, i], y_pred[:, i]))
    x_tick = n_steps_ahead + 1
    plt.plot(range(1, 6), r2_lag, 'o', c='#fde70c',
             mec='#8c8b8b', mew='1.5', ms=14)
    #plt.ylim([0, 1])
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(np.arange(1, x_tick, 1))
    plt.xlabel('Lags', fontsize=16)
    plt.ylabel(r'$R^2$', fontsize=16)
    plt.title(title, fontsize=18)
    print(r2_lag)


def error_boxplot(y_true, y_pred, n_steps_ahead, title):

    """

    param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: boxplots with the errors for each lags separately
    """

    error_dict = {}
    for i in range(n_steps_ahead):
        error_dict[i + 1] = ((y_true[:, i] - y_pred[:, i]) ** 2).flatten()

    fig, ax = plt.subplots()

    medianprops = dict(linestyle='-', linewidth=3.5, color='#fde70c')
    flierprops = dict(markeredgecolor='#8c8b8b')
    ax.boxplot(error_dict.values(), medianprops=medianprops,
               flierprops=flierprops)
    ax.set_xticklabels(error_dict.keys())
    title = title
    plt.title(title)
    plt.xlabel('Lags')
    plt.ylabel('Value')


def error_plot(y_true, y_pred, n_steps_ahead, title):

    """

    :param y_true: observed values
    :param y_pred: predicted values
    :param n_steps_ahead: prediction horizon
    :param title: title for the plot
    :return: error plot with time line for each step ahead seperate
    """

    mse_lag = []
    for i in range(n_steps_ahead):
        mse_lag.append(mean_squared_error(y_true[:, i], y_pred[:, i]))

    plt.plot(range(1, n_steps_ahead+1), mse_lag, 'o', c='#fde70c', markeredgecolor='#8c8b8b', markersize=14)
    plt.xticks(np.arange(1, n_steps_ahead + 1, 1))
    #plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('Lags')
    plt.ylabel('MSE')
    plt.title(title);
    print(mse_lag)
    #save_fig(figname)


def prediction_vs_observed_plot(compare, params, n_steps_ahead, y_train, y_test, title, trainplot=True):

    """

    :param compare: the models keys to compare i. e. 'lstm'
    :param params: models saved parameter
    :param n_steps_ahead: prediction horizon i. e. 5
    :param y_train: train set
    :param y_test: test set
    :param title: title of the plot
    :param trainplot: if it should be a plot for the training or testing performance
    :return: predicted vs. observed plot for each step ahean
    """


    fig = plt.figure(figsize=(10, 20))
    # if training set plot different
    if trainplot:
        x_vals = np.arange(len(y_train))
    else:
        x_vals = len(y_train) + np.arange(len(y_test))

    # iterate over the lags
    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i+1)
        # compare each value in each lag
        for key in compare:
            # if training set plot different
            if trainplot:
                y_vals = params[key]['pred_train'][:, i]
                label = params[key]['label'] + \
                    ' (train MSE: %.2e)' % params[key]['MSE_train steps ahead: '+str(i+1)]
            else:
                y_vals = params[key]['pred_test'][:, i]
                label = params[key]['label'] + \
                    ' (test MSE: %.2e)' % params[key]['MSE_test steps ahead:' + str(i+1)]

            plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=2)

        # if training set plot different
        if trainplot:
            plt.plot(x_vals, y_train[:, i], c="k", label="Observed", lw=2)

        else:
            plt.plot(x_vals, y_test[:, i],
                     c="k", label="Observed", lw=2)

        plt.xlim(x_vals.min(), x_vals.max())
        plt.xlabel('Time (ticks)', fontsize=14)
        plt.ylabel('$\hat{Y}$', rotation=0, fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.title(f'{i+1}' + title, fontsize=16)

    fig.tight_layout(pad=3.0)


def error_plot_timeline(compare, params, n_steps_ahead, y_train, y_test, title, trainplot=True):

    """

    :param compare: the models keys to compare i. e. 'lstm'
    :param params: models saved parameter
    :param n_steps_ahead: prediction horizon i. e. 5
    :param y_train: train set
    :param y_test: test set
    :param title: title of the plot
    :param trainplot: if it should be a plot for the training or testing performance
    :return: plot of the error timeline for each step ahean
    """

    fig = plt.figure(figsize=(10, 20))

    if trainplot:
        x_vals = np.arange(len(y_train))

    else:
        x_vals = len(y_train) + np.arange(len(y_test))

    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i + 1)
        for key in compare:
            if trainplot:
                y_vals = params[key]['pred_train'][:, i] - y_train[:, i]
                label = params[key]['label'] + ' (train MSE: %.2e)' % params[key][
                    'MSE_train steps ahead: ' + str(i + 1)]

            else:
                y_vals = params[key]['pred_test'][:, i] - y_test[:, i]
                label = params[key]['label'] + \
                    ' (test MSE: %.2e)' % params[key]['MSE_test steps ahead:' + str(i + 1)]
            plt.plot(x_vals, y_vals, c=params[key]['color'], label=label, lw=2)

        plt.axhline(0, linewidth=0.8)
        plt.xlim(x_vals.min(), x_vals.max())
        plt.xlabel('Time (ticks)', fontsize=16)
        plt.ylabel('$\hat{Y}-Y$', fontsize=16)
        plt.legend(loc="best", fontsize=12)
        plt.title(f'{i+1} ' + title, fontsize=16)

    fig.tight_layout(pad=3.0)


def plot_CV_histogram(results_df, key, ticker, n_steps, n_steps_ahead, uni):

    """

    :param results_df: results from cross-validation as pandas data frame
    :param key: the network to evaluate i. e. 'rnn'
    :param ticker: the ticker which was evaluated i. e. TMO
    :param n_steps: input lags, integer i. e. 10
    :param n_steps_ahead: prediction horizon, integer,  i. e. 5
    :param uni: if univariate or bivariate (with sentiment)
    :return: the cross validation performance plot
    """

    if uni:
        title = 'Univariate CV train and test scores of {} steps ahead of {} for {}'.format(n_steps, key, ticker)
        file_name = 'Histogramm_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)
    else:
        title = 'Mulitvariate CV train and test scores of {} steps ahead of {} for {}'.format(n_steps, key, ticker)
        file_name = 'Histogramm_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)

    plt.hist(results_df['mean_test_score'], label='mean test score')
    plt.hist(results_df['mean_train_score'], label='mean train score')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc=0, fontsize=16)
    save_fig(file_name)


def plot_post_distribution(t_post, ticker, n_steps, n_steps_ahead, uni):


    """

    :param t_post: possterier distribution
    :param ticker: name of asset i. e. TMO
    :param n_steps: input lags, integer, i. e. 10
    :param n_steps_ahead: prediction horizon, integer, i. e. 5
    :param uni: if uni or bivariate data, i.e. with sentiment
    :return: plot for models cross valdidation ditribution perforamnce
    """

    if uni:
        title = 'Posterior distribution for univariate of {} '.format(ticker)
        file_name = 'post_dist_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)
    else:
        title = 'Posterior distribution for mulivariate of {} '.format(ticker)
        file_name = 'post_dist_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)

    x = np.linspace(t_post.ppf(0.001), t_post.ppf(0.999), 100)
    plt.plot(x, t_post.pdf(x))
    plt.fill_between(x, t_post.pdf(x), 0, facecolor='#fde70c', alpha=.4)
    plt.ylabel('Probability density')
    plt.xlabel(r'Mean difference ($\mu$)')
    plt.title(title);

    save_fig(file_name)


def stock_price_plot(mu, sigma, n_steps, n_steps_ahead, y_true, y_pred, key, ticker, uni):

    """

    :param mu: mu of scaled data
    :param sigma: sigma of scaled data
    :param n_steps: input lags
    :param n_steps_ahead: prediction horizon
    :param y_true: true values
    :param y_pred: predicted values
    :param key: key of model i. e. 'lstm'
    :param ticker: ticker i. e. TMO
    :param uni: if uni- or bivariate data
    :return: plot with actual vs. predicted
    """



    fig = plt.figure(figsize=(5, 12))

    if uni:
        file_name = 'stock_price_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)
    else:
        file_name = 'stock_price_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(ticker, n_steps, n_steps_ahead)

    for i in range(n_steps_ahead):
        plt.subplot(n_steps_ahead, 1, i + 1)
        # just use the mu and sigma of the closing price
        test = pre.reverse_scaling(y_true[:, i], mu[0], sigma[0])
        pred = pre.reverse_scaling(y_pred[:, i], mu[0], sigma[0])

        title = 'Observed vs. predicted with {} lags and {} steps ahead of {} for {}'.format(
            n_steps, i + 1, key, ticker)
        plt.plot(test, label='Observed', linewidth=1.6)
        plt.plot(pred, label='Predicted', linewidth=1.6)
        plt.legend(loc=0, fontsize=14)
        plt.xlabel('Time (observations)', fontsize=8)
        plt.ylabel('Price in $', fontsize=8)
        plt.legend(loc="best", fontsize=8)
        plt.title(title, fontsize=12)

    save_fig(file_name)


def plot_train_val_loss(history, key, ticker, n_steps, n_steps_ahead, uni):

    """

    :param history: training history
    :param key: key of the funciton e. g. 'rnn'
    :param ticker: asset e. g. TMO
    :param n_steps: input lags, integer, e. g. 10
    :param n_steps_ahead: prediction horizon, e. .g. 5 (integer)
    :param uni: if uni-or bivariate data
    :return: learing curve plot (training and val loss)
    """


    if uni:
        file_name = '{}_train_val_los_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(key,ticker, n_steps, n_steps_ahead)
    else:
        file_name = '{}_train_val_los_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(key,ticker, n_steps, n_steps_ahead)

    title = 'Model loss of {} steps ahead of {} for {}'.format(n_steps, key, ticker)

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.xlabel('Epoch',fontsize=14)
    plt.legend(['Train', 'Val'], loc=0, fontsize=16)

    save_fig(file_name)


def scaled_observed_vs_predicted_plot(data1, data2, key, ticker, data_train, data_val,
                                      data_test, mu, sigma, params, n_steps, n_steps_ahead, uni):

    """

    :param data1: the splitted dataframe from the cut-off (special COVID-19 data cut-off) (first plit)
    :param data2: the splitted dataframe from the cut-off (special COVID-19 data cut-off) (second split)
    :param key: key for the model, i. e. 'rnn'
    :param ticker: assets ticker i. e. TMO
    :param data_train: training data split
    :param data_val: validaiton data spit
    :param data_test: testing data split
    :param mu: mu of the transformed data
    :param sigma: sigmma of the transformed data
    :param params: models funcitons params
    :param n_steps: input lags, integer, i.e. 10
    :param n_steps_ahead: prediction horizon, integer, i. e. 5
    :param uni: if bi- or univariate data to handle
    :return: scaled prediction plot
    """


    for i in range(n_steps_ahead):
        fig = plt.figure(figsize=(10, 20))
        plt.subplot(n_steps_ahead, 1, i + 1)

        train = pre.reverse_scaling(
            params[key]['pred_train'][:, i], mu[0], sigma[0])
        val = pre.reverse_scaling(
            params[key]['pred_val'][:, i], mu[0], sigma[0])
        test = pre.reverse_scaling(
            params[key]['pred_test'][:, i], mu[0], sigma[0])
        title = 'Observed vs. predicted with {} lags and {} steps ahead of {} for {}'.format(
            n_steps, i + 1, key, ticker)

        look_back = (n_steps + n_steps_ahead)-1
        data3 = pd.concat([data1, data2])

        if uni:
            file_name = '{}_stock_price_plot_price_{}_n_steps_{}_n_steps_ahead_{}'.format(key, ticker, n_steps,
                                                                                          i+1)
        else:
            file_name = '{}_stock_price_plot_price_conv_{}_n_steps_{}_n_steps_ahead_{}'.format(key, ticker, n_steps,
                                                                                               i+1)

        plt.plot(data1.close, 'k', label="observed", lw=1.5)
        plt.plot(data2.close, 'k', lw=1.5)
        plt.plot(data_train.index[look_back:], train,
                 c='#767a76', lw=1.5, label="train")
        plt.plot(data_val.index[look_back:], val,
                 lw=1.5, c='#12a506', label="val")
        plt.plot(data_test.index[look_back:], test,
                 lw=1.5, c='#fde70c', label="test")
        plt.fill_between(
            [data_train.index[-1], data_val.index[0]], 0, np.max(data3.close))
        plt.text((data_val.index[5]), 40, "COVID-19", fontsize=18)
        plt.xlabel('Date')
        plt.ylabel('Price in $')
        plt.title(title, fontsize=16)
        plt.legend(fontsize=16)

        save_fig(file_name)


