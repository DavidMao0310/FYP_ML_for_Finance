import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import KernelPCA, PCA
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

bata =data
ret = data.pct_change(1)
ret.dropna(inplace=True)
hs_ret = hs.pct_change(1)
hs_ret.dropna(inplace=True)
fig = plt.figure(figsize=(14, 6))


def plotreturn(hs_ret, ret):
    plt.plot(hs_ret.cumsum(), label='CSI300 Index return', color='black')
    plt.plot(ret.cumsum(), alpha=0.4, linestyle='-')
    plt.legend()
    plt.title('cumulative return')
    fig.tight_layout()
    plt.savefig('Plots/cum_return.png')
    plt.show()


plotreturn(hs_ret=hs_ret,ret=ret)

data = ret
names = data.columns.tolist()

codes = np.array(data.columns)
variation = data.dropna().values
edge_model = covariance.GraphicalLassoCV()
X = variation.copy()
X /= X.std(axis=0)
edge_model.fit(X)
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=None)
n_labels = labels.max()

group = []
for i in range(n_labels + 1):
    group.append(codes[labels == i].tolist())


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


def PCAfilter(gx):
    gxlist = reverse(gx)
    score = []
    for k in gxlist:
        gdata = data[k]
        print(gdata)
        scale_function = lambda x: (x - x.mean()) / x.std()
        pca = KernelPCA().fit(gdata.apply(scale_function))
        print(getweight(pca.lambdas_))
        score.append(getweight(pca.lambdas_))

    return score


def heat(gx):
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(data[gx].corr(), cmap='coolwarm')
    fig.tight_layout()
    plt.show()


def plot_singal_return(gx):
    scoredict = {}
    for k in gx:
        df = data[[str(k)]]
        sc = pd.concat([df, hs_ret['hs_close']], axis=1)
        sc['score'] = np.sign(sc[str(k)].cumsum() - sc['hs_close'].cumsum())
        scoredict[k] = sc['score'].sum()
        plt.plot(df.cumsum(), alpha=0.5, linestyle='-', label=str(k))
    m = max(scoredict.items(), key=lambda x: x[1])[0]
    plt.plot(data[m].cumsum(), linestyle=':', label=None,color='black')
    plt.plot(hs_ret.cumsum(), label='CSI300 Index return', color='black')
    plt.legend()
    plt.title('cumulative return group'+str(group.index(gx)))
    return m


select = []
fig = plt.figure(figsize=(30,20))
fig.add_subplot(321)
sns.heatmap(data.corr(),cmap='coolwarm')
plt.title('correlation of stocks before clustering')
for i in group:
    j = 322+group.index(i)
    fig.add_subplot(int(j))
    select.append(plot_singal_return(i))
fig.tight_layout()
plt.savefig('Plots/culstering.png')
plt.show()
print(select)

bata = bata[select]
bata.to_csv('dataset/portfoliodata.csv')
