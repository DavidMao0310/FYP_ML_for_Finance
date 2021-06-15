import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import KernelPCA, PCA
from sklearn import cluster, covariance, manifold

plt.style.use('ggplot')


def read(df):
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df


def sharpe_ratio(return_series, N=252, rf=0.0313):
    mean = return_series.mean() * N - rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma


hs = pd.read_csv('dataset/hsindex.csv')
data = pd.read_csv('dataset/pricedata.csv')
data = read(data)
hs = read(hs)

ret = data.pct_change(1)
ret.dropna(inplace=True)
hs_ret = hs.pct_change(1)
hs_ret.dropna(inplace=True)

data = ret
names = data.columns.tolist()
print(data)

N = 252  # 252 trading days in a year
rf = 0.0313  # risk free rate


def getweight(x):
    y = x / x.sum()
    return y


def reverse(x):
    z = []
    y = x
    for i in range(len(x)):
        last = [y.pop()]
        y = last + y
        z.append(y.copy())
    return z


def KPCAfilter(gx):
    gxlist = reverse(gx)
    score = {}
    for k in gxlist:
        gdata = data[k]
        scale_function = lambda x: (x - x.mean()) / x.std()
        pca = KernelPCA().fit(gdata.apply(scale_function))
        w = getweight(pca.lambdas_)
        sp = sharpe_ratio(np.array(data.apply(lambda x: np.sum(np.dot(x, w)), axis=1)))
        score[sp]=[k,w]

    return score


print(list(KPCAfilter(names).keys()))
print(KPCAfilter(names)[max(list(KPCAfilter(names).keys()))])

def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()


full_df = pd.read_csv('dataset/pricedata.csv')
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
rets2['KPCA'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsso)), axis=1))
rets = pd.concat([rets, rets2], axis=1)
print(rets)

N = 252  # 252 trading days in a year
rf = 0.0313  # risk free rate
fig = plt.figure(figsize=(14, 8))
sharpes = rets.apply(sharpe_ratio, args=(N, rf,), axis=0)
fig.add_subplot(221)
sharpes.plot.bar()


def sortino_ratio(series, N, rf):
    mean = series.mean() * N - rf
    std_neg = series[series < 0].std() * np.sqrt(N)
    return mean / std_neg


sortinos = rets.apply(sortino_ratio, args=(N, rf,), axis=0)
fig.add_subplot(222)
sortinos.plot.bar()
plt.ylabel('Sortino Ratio')


def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()


max_drawdowns = rets.apply(max_drawdown, axis=0)
fig.add_subplot(223)
max_drawdowns.plot.bar()
plt.ylabel('Max Drawdown')

calmars = rets.mean() * 252 / abs(max_drawdowns)
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
plt.figure(figsize=(12, 6))
plt.plot(rets.cumsum(), alpha=0.8)
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = hs.pct_change(1).dropna()
plt.plot(hs_return.cumsum(), color='black', label='CSI300 Index')
plt.table(cellText=np.round(btstats.values, 2), colLabels=btstats.columns,
          rowLabels=btstats.index, rowLoc='center', cellLoc='center', loc='top',
          colWidths=[0.25] * len(btstats.columns))
plt.legend()
plt.title('cumulative return')
plt.tight_layout()
plt.show()
