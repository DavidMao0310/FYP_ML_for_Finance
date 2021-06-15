import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold
plt.style.use('ggplot')

data = pd.read_csv('dataset/pricedata.csv')
data.set_index('trade_date', inplace=True)
data = data.set_index(pd.to_datetime(data.index))
ret = data.pct_change(1)
ret.dropna(inplace=True)

fig = plt.figure(figsize=(14, 5))
fig.add_subplot(121)
plt.plot(data,alpha=0.6)
plt.title('Close Price')
fig.add_subplot(122)
plt.plot(ret,alpha=0.6)
plt.title('Return')
fig.tight_layout()
plt.show()
data = ret
names = data.columns.tolist()


codes = np.array(data.columns)
variation=data.dropna().values
edge_model = covariance.GraphicalLassoCV()
X = variation.copy()
X /= X.std(axis=0)
edge_model.fit(X)
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=None)
n_labels = labels.max()

group = []
for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(codes[labels == i])))
    group.append(codes[labels == i].tolist())

print(group)


node_position_model= manifold.LocallyLinearEmbedding(n_components=2,eigen_solver='dense',n_neighbors=6)

embedding=node_position_model.fit_transform(X.T).T
plt.figure(1,facecolor='grey',figsize=(16,8))
plt.clf()
ax=plt.axes([0.,0.,1.,1.])
plt.axis('off')

partial_correlations=edge_model.precision_.copy()
d=1/np.sqrt(np.diag(partial_correlations))
partial_correlations*=d
partial_correlations*=d[:,np.newaxis]
non_zero=(np.abs(np.triu(partial_correlations,k=1))>0.02)

plt.scatter(embedding[0],embedding[1],s=100*d**2,c=labels,cmap=plt.cm.nipy_spectral)

start_idx,end_idx=np.where(non_zero)
segments=[[embedding[:,start],embedding[:,stop]]
          for start,stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc=LineCollection(segments,zorder=0,cmap=plt.cm.hot_r,
                  norm=plt.Normalize(0,.7*values.max()),colors='grey')

lc.set_array(values)
lc.set_linewidths(20*values)
ax.add_collection(lc)
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .01
    else:
        horizontalalignment = 'right'
        x = x - .01
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .01
    else:
        verticalalignment = 'top'
        y = y - .01
    plt.text(x, y, name, size=8,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                       alpha=.6))


plt.xlim(embedding[0].min()-.15*embedding[0].ptp(),
embedding[0].max()+.10*embedding[0].ptp(),)
plt.ylim(embedding[1].min()-.03*embedding[1].ptp(),
embedding[1].max()+.03*embedding[1].ptp())
plt.show()