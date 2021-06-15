import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from arch import arch_model

warnings.simplefilter('ignore')
plt.style.use('ggplot')
rnnvol = pd.read_csv('dataset/new_rnn_vol.csv')
use_data = pd.read_csv('dataset/conditional_data.csv')
use_data['Date'] = pd.to_datetime(use_data['Date'])
use_data.set_index('Date', inplace=True)
print(use_data)

def readdata(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


rnnvol = readdata(rnnvol)
rnnvol = rnnvol[['vol']]*100

full = rnnvol
print(full)


R = use_data['return']
R = R
#####R is return series. %


# Specify and fit a GARCH model
split_date = '2020-01-31'
basic_gm = arch_model(R, vol='Garch', p=1, o=0, q=1, dist='skewt')
res = basic_gm.fit(disp='off', last_obs=split_date)

forcast_first_observation_date = '2020-02-01'
# Make variance forecast
forecasts = res.forecast(start=forcast_first_observation_date)

cond_mean = forecasts.mean[forcast_first_observation_date:]
full = full[forcast_first_observation_date:]
cond_var = np.array(full['vol'])

q = basic_gm.distribution.ppf([0.01, 0.05], res.params[-2:])

print('1% parametric quantile: ', q[0], '\n', '5% parametric quantile: ', q[1])
# Calculate the VaR

VaR_parametric = cond_mean.values.ravel() + cond_var * q[1]
VaR_parametric2 = cond_mean.values.ravel()+ cond_var * q[0]



# Save VaR in a DataFrame
VaR_parametric = pd.DataFrame(VaR_parametric, columns=['5%'], index=full.index)
VaR_parametric2 = pd.DataFrame(VaR_parametric2, columns=['1%'], index=full.index)
# Plot the VaR
plt.figure(figsize=(12, 6))
plt.plot(VaR_parametric, color='chocolate', label='1-day 95% VaR', alpha=0.7)
plt.plot(VaR_parametric2, color='rosybrown', label='1-day 99% VaR', alpha=0.7)
plt.scatter(full.index, R[forcast_first_observation_date:], color='green',
            label='Portfolio Daily Returns', alpha=0.7, marker='o')
c = []
rets_use = R[split_date:].copy()
cond1 = rets_use < VaR_parametric2['1%']
cond2 = rets_use < VaR_parametric['5%']
length = len(R[forcast_first_observation_date:])
print('95%:',len(rets_use[cond2]),len(rets_use[cond2])/length,'\n 99%:',
      len(rets_use[cond1]),len(rets_use[cond1])/length)
plt.scatter(rets_use[cond2].index, rets_use[cond2], color='blue', marker='s',
            label='95% Exceedance')
plt.scatter(rets_use[cond1].index, rets_use[cond1], color='red', marker='s',
            label='99% Exceedance')
plt.legend()
plt.title('LSTM VaR')
plt.tight_layout()
plt.show()