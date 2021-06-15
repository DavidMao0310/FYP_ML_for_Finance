import pandas as pd
import numpy as np
import math
import talib as ta
import scipy.optimize as sco
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

plt.style.use('ggplot')


def read(df):
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df


full_df = pd.read_csv('dataset/pricedata.csv')
full_df = read(full_df)
r_f = 0.0313
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = hs.pct_change(1).dropna()
returns_daily = full_df.pct_change()
rets = returns_daily.dropna()
data = (rets - rets.mean()) / rets.std()

stock_tickers = data.columns.values
n_tickers = len(stock_tickers)
# Dividing the dataset into training and testing sets
percentage = int(len(data) * 0.8)
X_train = data[:percentage]
X_test = data[percentage:]

X_train_raw = rets[:percentage]
X_test_raw = rets[percentage:]
cov_matrix = X_train.cov()
pca = PCA()
pca.fit(cov_matrix)


def plotPCA(plot=False):
    # Visualizing Variance against number of principal components.
    cov_matrix_raw = X_train_raw.loc[:, X_train_raw.columns != 'DJIA'].cov()

    var_threshold = 0.95
    var_explained = np.cumsum(pca.explained_variance_ratio_)
    num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1
    if plot:
        print('%d principal components explain %.2f%% of variance' % (num_comp, 100 * var_threshold))

        # PCA percent variance explained.
        bar_width = 0.9
        n_asset = stock_tickers.shape[0]
        x_indx = np.arange(n_asset)
        fig, ax = plt.subplots(figsize=(12, 4))
        # Eigenvalues measured as percentage of explained variance.
        rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width)
        ax.set_xticks(x_indx + bar_width / 2)
        ax.set_xticklabels(list(range(n_asset)), rotation=45)
        ax.set_title('Percent variance explained')
        ax.set_ylabel('Explained Variance')
        ax.set_xlabel('Principal Components')
        plt.tight_layout()
        plt.savefig('pcavariance.png')
        plt.show()


plotPCA(plot=True)

projected = pca.fit_transform(cov_matrix)
pcs = pca.components_
print(pcs)

# Sharpe Ratio
def sharpe_ratio(ts_returns, periods_per_year=252):
    '''
    Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
    It calculares the annualized return, annualized volatility, and annualized sharpe ratio.
    ts_returns are  returns of a signle eigen portfolio.
    '''
    n_years = ts_returns.shape[0] / periods_per_year
    annualized_return = np.power(np.prod(1 + ts_returns), (1 / n_years)) - 1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe


def optimizedPortfolio():
    n_portfolios = len(pcs)
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    highest_sharpe = 0

    for i in range(n_portfolios):
        pc_w = pcs[:, i] / sum(pcs[:, i])
        eigen_prtfi = pd.DataFrame(data={'weights': pc_w.squeeze()}, index=stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)

        eigen_prti_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen_prtfi / n_portfolios)
        eigen_prti_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_test.index)
        er, vol, sharpe = sharpe_ratio(eigen_prti_returns)
        annualized_ret[i] = er
        annualized_vol[i] = vol
        sharpe_metric[i] = sharpe

    # find portfolio with the highest Sharpe ratio
    highest_sharpe = np.argmax(sharpe_metric)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' %
          (highest_sharpe,
           annualized_ret[highest_sharpe] * 100,
           annualized_vol[highest_sharpe] * 100,
           sharpe_metric[highest_sharpe]))

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 2)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Portfolios')

    results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
    results.dropna(inplace=True)
    results.sort_values(by=['Sharpe'], ascending=False, inplace=True)

    plt.show()


optimizedPortfolio()


def PCWeights():
    '''
    Principal Components (PC) weights for each 28 PCs
    '''
    weights = pd.DataFrame()

    for i in range(len(pcs)):
        weights["weights_{}".format(i)] = pcs[:, i] / sum(pcs[:, i])

    weights = weights.values.T
    return weights


weights = PCWeights()
print(weights)
portfolio = pd.DataFrame()


def plotEigen(weights, plot=False, portfolio=portfolio):
    portfolio = pd.DataFrame(data={'weights': weights.squeeze()}, index=stock_tickers)
    portfolio.sort_values(by=['weights'], ascending=False, inplace=True)

    if plot:
        print('Sum of weights of current eigen-portfolio: %.2f' % np.sum(portfolio))
        portfolio.plot(title='Current Eigen-Portfolio Weights',
                       figsize=(12, 6),
                       xticks=range(0, len(stock_tickers), 1),
                       rot=45,
                       linewidth=3
                       )
        plt.tight_layout()
        plt.show()

    return portfolio


# Weights are stored in arrays, where 0 is the first PC's weights.
a = np.array(plotEigen(weights=weights[2], plot=True)['weights'].tolist()).round(4)
print(a.tolist())


def plotSharpe(eigen):
    '''

    Plots Principle components returns against real returns.

    '''

    eigen_portfolio_returns = np.dot(X_test_raw.loc[:, eigen.index], eigen / len(pcs))
    eigen_portfolio_returns = pd.Series(eigen_portfolio_returns.squeeze(), index=X_test.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)
    print('Current Eigen-Portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (
    returns * 100, vol * 100, sharpe))
    year_frac = (eigen_portfolio_returns.index[-1] - eigen_portfolio_returns.index[0]).days / 252



plotSharpe(eigen=plotEigen(weights=weights[2]))

