import pandas as pd
import numpy as np
import talib as ta
import scipy.optimize as sco
from matplotlib import pyplot as plt
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)


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
rets2 = pd.DataFrame(index=rets.index)
rets2['max sharpe ratio'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightss)), axis=1))
rets2['lowest volatility'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsv)), axis=1))
rets2['highest return'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsr)), axis=1))
rets2['max sortino ratio'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsso)), axis=1))
rets = pd.concat([rets,rets2],axis=1)
print(rets)

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

N = 252 #252 trading days in a year
rf =0.0313 # risk free rate
fig = plt.figure(figsize=(14,8))
sharpes = rets.apply(sharpe_ratio, args=(N,rf,),axis=0)
fig.add_subplot(221)
sharpes.plot.bar()
plt.ylabel('Sharpe Ratio')

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg


sortinos = rets.apply(sortino_ratio, args=(N,rf,), axis=0 )
fig.add_subplot(222)
sortinos.plot.bar()
plt.ylabel('Sortino Ratio')

def max_drawdown(return_series):
    comp_ret = (return_series+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()


max_drawdowns = rets.apply(max_drawdown,axis=0)
fig.add_subplot(223)
max_drawdowns.plot.bar()
plt.ylabel('Max Drawdown')

calmars = rets.mean()*252/abs(max_drawdowns)
fig.add_subplot(224)
calmars.plot.bar()
plt.ylabel('Calmar ratio')
fig.tight_layout()
plt.show()

btstats = pd.DataFrame()
btstats['sortino'] = sortinos
btstats['sharpe'] = sharpes
btstats['maxdd'] = max_drawdowns
btstats['calmar'] = calmars
plt.figure(figsize=(12,6))
plt.plot(rets.cumsum(),alpha=0.8)
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = hs.pct_change(1).dropna()
plt.plot(hs_return.cumsum(),color='black',label='CSI300 Index')
plt.table(cellText=np.round(btstats.values,2), colLabels=btstats.columns,
          rowLabels=btstats.index,rowLoc='center',cellLoc='center',loc='top',
          colWidths=[0.25]*len(btstats.columns))
plt.legend()
plt.title('cumulative return')
plt.tight_layout()
plt.show()