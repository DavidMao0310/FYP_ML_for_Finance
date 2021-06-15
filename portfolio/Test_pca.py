import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import KernelPCA,PCA
from sklearn import cluster, covariance, manifold

plt.style.use('ggplot')


def read(df):
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    return df


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
    score = []
    for k in gxlist:
        gdata = data[k]
        scale_function = lambda x: (x - x.mean()) / x.std()
        pca = KernelPCA().fit(gdata.apply(scale_function))
        w = getweight(pca.lambdas_)
        score.append(getweight(pca.lambdas_))

    return score


KPCAfilter(names)
