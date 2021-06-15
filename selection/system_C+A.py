from canslim import *
import pandas as pd

pd.set_option('display.max_columns', None)
# We need collect all stock data then make them into a single dataframe which is tracking by symbol
datac = pd.read_excel('company_consumption_details_data.xlsx')
datat = pd.read_excel('company_tech_details_data.xlsx')
datam = pd.read_excel('company_medical_details_data.xlsx')
RPS = pd.read_csv('RPS_selection_result.csv')
com=RPS['L_consumption'].values.tolist()
tech=RPS['L_tech'].values.tolist()
med=RPS['L_medical'].values.tolist()

def Cfilter(df, EPSgrowth_prequa, EPSgrowth_preyear_samequa, EPS_increasly_acc):
    '''

    :param df: dataframe contains 'EPS basic' columns
    :param EPSgrowth_prequa: EPS growth compared to previous quarters
    :param EPSgrowth_preyear_samequa: EPS growth compared to same quarters previous year
    :param EPS_increasly_acc: EPS increasly accelerating for at least quarters
    :return: CANSLIM_C-filter
    '''

    ac = AnnualChange(cols=['eps'])
    df = ac.fit_transform(df)
    qc = QuarterlyChange(cols=['eps'])
    df = qc.fit_transform(df)
    ag = AccelIncrease(cols=['eps_qperc'])
    df = ag.fit_transform(df)
    # EPS basic: EPS value
    # EPS basic_qchange: Quartly change in EPS
    # EPS basic_annual: Annually change in EPS
    # EPS basic_qperc: EPS compared to previous quarters
    EPS_filter = (df['eps_qperc'] >= EPSgrowth_prequa) & (df['eps_annualp'] >= EPSgrowth_preyear_samequa) & \
                 (df['eps_qperc_accstreak'] >= EPS_increasly_acc)
    return EPS_filter



def Afilter(df, ROE_growth):
    '''

    :param df: dataframe contains 'Earnings' and 'ROE' columns
    :param Earing_growth_per: Annual earnings growth should be at least $ percentage
    :param Earing_growth_year_need: Annual earnings growth should be at least $ percentage over $ years
    :param ROE_growth: Annual Return on Equity should be more than $ percentage
    '''
    df['roe_growth']=df['roe'].pct_change(1)
    A_filter1 = (df['roe_growth'] > ROE_growth)

    return A_filter1

def AC_select(stocks):
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks['Year'] = stocks['date'].dt.year
    stocks['Month'] = stocks['date'].dt.month
    C_filter = Cfilter(stocks, 0.1, 0.1, 1)
    A_filter = Afilter(stocks, ROE_growth=0.1)
    AC_filter = C_filter & A_filter
    AC_select_stocks = stocks[AC_filter]
    AC_select_stocks_end = AC_select_stocks[AC_select_stocks['Year'] >= max(stocks['date']).year - 1]
    AC_select_stocks_end.reset_index(inplace=True)
    #list
    return AC_select_stocks_end['symbol'].drop_duplicates(keep=False).values

ac_com = AC_select(datac)
ac_tech = AC_select(datat)
ac_med = AC_select(datam)

f_com = list(set(ac_com).intersection(com))
f_tech = list(set(ac_tech).intersection(tech))
f_med = list(set(ac_med).intersection(med))
f = f_med+f_tech+f_com
print(f)
final = pd.DataFrame()


final['stock'] = np.array(f)

print(final)

final.to_csv('CANSLIM_selection_result.csv')






