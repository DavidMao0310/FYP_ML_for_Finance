import pandas as pd
import numpy as np
import talib as ta
import scipy.optimize as sco
from matplotlib import pyplot as plt

plt.style.use('ggplot')


def read(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()


full_df = pd.read_csv('dataset/portfoliodata.csv')
full_df = read(full_df)
r_f = 0.0313
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = hs.pct_change(1).dropna()
returns_daily = full_df.pct_change()
rets = returns_daily.dropna()
weightss = [3.73080531e-18, 2.15451476e-01, 7.84548524e-01, 5.58136317e-17, 1.72668439e-16]
weightsv = [0.33589537, 0.03380231, 0.26294063, 0.25046491, 0.11689677]
weightsr = [0.00000000e+00, 3.33066907e-15, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00]
weightsso = [0., 0.26355297, 0.73410166, 0.00234538, 0.]


def lookcal(ret, weights, name):
    df = ret.copy()
    df[str(name) + 'portret'] = np.array(df.apply(lambda x: np.sum(np.dot(x, weights)), axis=1))
    print(str(name) + '_mdd', max_drawdown(df[str(name) + 'portret']))
    calmar_ratio = df[str(name) + 'portret'].mean() * 252 / abs(max_drawdown(df[str(name) + 'portret']))
    print(str(name) + '_calmar ratio', calmar_ratio)


lookcal(rets, weightss, 'sharpe')
lookcal(rets, weightsv, 'vola')
lookcal(rets, weightsr, 'ret')
lookcal(rets, weightsso, 'sort')
for name in rets.columns.tolist():
    print(str(name) + '_mdd', max_drawdown(rets[str(name)]))
    calmar_ratio = rets[str(name)].mean() * 252 / abs(max_drawdown(rets[str(name)]))
    print(str(name) + '_calmar ratio', calmar_ratio)

