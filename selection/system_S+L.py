import pandas as pd
from datetime import timedelta, datetime
import numpy as np
from datetime import datetime
from datetime import timedelta

pd.set_option('display.max_columns', None)
datac = pd.read_csv('testrp_con_data.csv')
datat = pd.read_csv('testrp_tech_data.csv')
datam = pd.read_csv('testrp_med_data.csv')


def cal_ret(df, w):
    '''w:week5; month:20; semi-annual:120; annual:250
    '''
    df = df / df.shift(w) - 1
    return df.iloc[w:, :].fillna(0)


def get_RPS(ser):
    df = pd.DataFrame(ser.sort_values(ascending=False))
    df['n'] = range(1, len(df) + 1)
    df['rps'] = (1 - df['n'] / len(df)) * 100
    return df


def all_RPS(data):
    dates = data.index
    dates1 = [x for x in dates]
    st = dates[0]
    end = dates[-1]
    RPS = {}
    for i in range(len(data)):
        RPS[dates[i]] = pd.DataFrame(get_RPS(data.iloc[i]).values,
                                     columns=['Rate of Return', 'Rank', 'RPS'],
                                     index=get_RPS(data.iloc[i]).index)
    return RPS, dates1, st, end


def all_data(rps, ret):
    df = pd.DataFrame(np.NaN, columns=ret.columns, index=ret.index)
    for date in ret.index:
        d = rps[date]
        for c in d.index:
            df.loc[date, c] = d.loc[c, 'RPS']
    return df


def Lfilter(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.set_index('date', inplace=True)
    df.index = (pd.to_datetime(df.index)).strftime('%Y-%m-%d')
    df.dropna(inplace=True, axis=1)

    window = 120
    ret_use = cal_ret(df, w=window)
    rps_use, rps_dates, rps_start, rps_end = all_RPS(ret_use)
    dates_list = [datetime.strftime(x, '%F') for x in pd.date_range(rps_start, rps_end, freq='m')]
    df_rps_rank = pd.DataFrame()
    iii=1
    for date in dates_list:
        print(iii)
        iii+=1
        while date not in rps_dates:
            dt = datetime.strptime(date, '%Y-%m-%d')
            date = dt - timedelta(days=1)
            date = date.strftime('%Y-%m-%d')
        df_rps_rank[date] = rps_use[date].index[:50]
    L_select_stocks = df_rps_rank.iloc[:, -1].values.tolist()
    # L select can get the strength candidates
    return L_select_stocks


L_select_stocks1 = Lfilter(datac)
L_select_stocks2 = Lfilter(datat)
L_select_stocks3 = Lfilter(datam)

final = pd.DataFrame()
final['L_consumption'] = np.array(L_select_stocks1)
final['L_tech'] = np.array(L_select_stocks2)
final['L_medical'] = np.array(L_select_stocks3)
print(final)
final.to_csv('RPS_selection_result.csv')

