import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch.unitroot import ADF

plt.style.use('ggplot')


def read(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


full_df = pd.read_csv('dataset/portfoliodata.csv')
full_df = read(full_df)
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = hs.pct_change(1).dropna()
returns_daily = full_df.pct_change()
rets = returns_daily.dropna()
rets = pd.concat([rets, hs_return], axis=1)
weightss = [3.73080531e-18, 2.15451476e-01, 7.84548524e-01, 5.58136317e-17, 1.72668439e-16, 0]
weightsv = [0.33589537, 0.03380231, 0.26294063, 0.25046491, 0.11689677, 0]
weightsr = [0.00000000e+00, 3.33066907e-15, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0]
weightsso = [0., 0.26355297, 0.73410166, 0.00234538, 0., 0]
rets2 = pd.DataFrame(index=rets.index)
rets2['max sharpe ratio'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightss)), axis=1))
rets2['lowest volatility'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsv)), axis=1))
rets2['highest return'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsr)), axis=1))
rets2['max sortino ratio'] = np.array(rets.apply(lambda x: np.sum(np.dot(x, weightsso)), axis=1))
rets = pd.concat([rets, rets2], axis=1)


def make_MASM(data, name):
    df_ret = data
    df_ret.plot(title=str(name) + '_Rate of Return', figsize=(12, 4), color='forestgreen')
    plt.show()
    # ADF Test
    adf = ADF(df_ret)
    print('ADF test result \n', adf)
    # Fit MSAM model
    mod = sm.tsa.MarkovRegression(df_ret.dropna(),
                                  k_regimes=3, trend='nc', switching_variance=True)
    res = mod.fit()
    print(res.summary())

    fig, axes = plt.subplots(3, figsize=(12, 8))
    ax = axes[0]
    ax.plot(res.smoothed_marginal_probabilities[0], color='skyblue')
    ax.set(title=str(name) + '_Low volatility smoothed probability graph')
    ax = axes[1]
    ax.plot(res.smoothed_marginal_probabilities[1], color='darkseagreen')
    ax.set(title=str(name) + '_Middle volatility smoothed probability graph')
    ax = axes[2]
    ax.plot(res.smoothed_marginal_probabilities[2], color='sandybrown')
    ax.set(title=str(name) + '_High volatility smoothed probability graph')
    fig.tight_layout()
    plt.show()


for i in rets.columns.tolist():
    print(i)
    try:
        make_MASM(rets[str(i)], name='CSI300 Index')
    except:
        pass

