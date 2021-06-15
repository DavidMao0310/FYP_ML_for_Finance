import pandas as pd
import numpy as np
import math
import talib as ta
import scipy.optimize as sco
from matplotlib import pyplot as plt
plt.style.use('ggplot')
weights = [3.73080531e-18, 2.15451476e-01, 7.84548524e-01, 5.58136317e-17, 1.72668439e-16]

def read(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

full_df = pd.read_csv('dataset/portfoliodata.csv')
full_df = read(full_df)

def get_portret(df,weights):
    R = np.log(df / df.shift(1))  # get log return
    R = R.dropna()
    R['return'] = R.apply(lambda x: np.sum(np.dot(x, weights)), axis=1)
    return R

full_df = get_portret(full_df,weights)
print(full_df)
hs = pd.read_csv('dataset/hsindex.csv')
hs['trade_date'] = pd.to_datetime(hs['trade_date'])
hs.set_index('trade_date', inplace=True)
hs_return = np.log(hs / hs.shift(1)).dropna()

plt.figure(figsize=(12,6))
full_df['return'].cumsum().plot(figsize=(12,6))
plt.plot(hs_return.cumsum(),color='black',label='CSI300 Index')
plt.legend()
plt.title('portfolio cumulative return')
plt.tight_layout()
plt.show()
#full_df.to_csv('dataset/portret.csv')