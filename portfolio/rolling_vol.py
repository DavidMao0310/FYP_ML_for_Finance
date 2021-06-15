import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import talib as ta

warnings.simplefilter('ignore')
from arch import arch_model

plt.style.use('ggplot')


def read(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


full_df = pd.read_csv('dataset/portret.csv')
full_df = read(full_df)
R = full_df['return']
R = 100 * R
vol = ta.STDDEV(R,timeperiod=15).dropna()
print(vol)
vol.to_csv('dataset/rollingvol.csv')