import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import armagarch as ag
import warnings

warnings.simplefilter('ignore')
plt.style.use('fivethirtyeight')

data = pd.read_csv('dataset/portret.csv')


def datapre(data):
    data.set_index('Date', inplace=True)
    data = data.set_index(pd.to_datetime(data.index))
    data.dropna(inplace=True)
    data = 100 * data[['return']]
    return data


data = datapre(data)


def make_armagarch_dataset(df, armaorder=(1, 1), garchorder=(1, 1), dist='T', name='required'):
    meanMdl = ag.ARMA(order={'AR': armaorder[0], 'MA': armaorder[1]})
    volMdl = ag.garch(order={'p': garchorder[0], 'q': garchorder[1]})
    if dist == 'T':
        dist = ag.tStudent()
    elif dist == 'N':
        dist = ag.normalDist()
    else:
        print('Error')
        pass
    # set-up the model
    model = ag.empModel(df, meanMdl, volMdl, dist)
    model.fit()
    # get the conditional mean
    cm = model.Ey
    # get conditional sigma
    ht = model.ht
    cs = np.sqrt(ht)
    full = pd.concat([df, cs], axis=1)
    return full


# fit_ARMAGARCH
fulldata = make_armagarch_dataset(data, armaorder=(2, 1), garchorder=(1, 1),
                                  dist='T', name='port_data')
print(fulldata)
fulldata.to_csv('dataset/conditional_data.csv')
