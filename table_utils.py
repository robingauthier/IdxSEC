import re
import pandas as pd
import numpy as np

pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 300
pd.set_option('future.no_silent_downcasting', True)

def filter_dimension_dataframe(df_pandas,max=2):
    if df_pandas is None:
        return None
    if len(df_pandas.shape) <= 1:
        return None
    if df_pandas.shape[1] <= 1:
        return None
    if df_pandas.shape[0] < max:
        return None
    df_pandas = df_pandas.replace('', np.nan)
    df_pandas = df_pandas.replace('\u200b', np.nan)
    df_pandas = df_pandas.dropna(axis=0, how='all')
    df_pandas = df_pandas.dropna(axis=1, how='all')
    if len(df_pandas.shape) <= 1:
        return None
    if df_pandas.shape[1] <= 1:
        return None
    if df_pandas.shape[0] < max:
        return None
    return df_pandas

def remove_nan_columns(df_pandas,debug=0,threshold=0.9):
    df_pandas=df_pandas.dropna(axis=1,how='all')
    return df_pandas

def clean_nan_col_rows(df_pandas):
    return df_pandas.replace('', np.nan).dropna(how='all', axis=1).dropna(how='all', axis=0)


def remove_empty_columns(df_pandas,debug=0,threshold=0.9):
    keepidx = []
    delidx = []

    for i in range(df_pandas.shape[1]):
        vsserie = df_pandas.iloc[:, i].fillna('').astype(str)
        if debug>0:
            print(i)
            print(vsserie)
            print(np.all(vsserie==''))
            print('-'*10)
        if not np.all(vsserie==''):
            keepidx+=[i]
        else:
            delidx+=[i]
    if len(delidx)>0:
        return df_pandas.iloc[:,list(set(keepidx))]
    return df_pandas

def cut_content_for_display(df_pandas,n_cut=20):
    ndf_pandas=df_pandas.copy()
    for i in range(df_pandas.shape[1]):
        if not ndf_pandas.iloc[:,i].dtype=='O':
            continue
        ndf_pandas.iloc[:,i]=ndf_pandas.iloc[:,i].astype(str).str[:n_cut]
    return ndf_pandas


def remove_duplicated_columns(df_pandas,debug=0,threshold=0.9):
    delidx=[]
    keepidx=[0]
    for i in range(1,df_pandas.shape[1]):
        refserie =df_pandas.iloc[:,i-1].fillna('').astype(str)
        vsserie = df_pandas.iloc[:, i].fillna('').astype(str)
        ratio=(refserie==vsserie).sum()/refserie.shape[0]
        refserie =df_pandas.iloc[:, i-1].fillna(0)
        vsserie = df_pandas.iloc[:, i].fillna(0)
        ratio2=(refserie==vsserie).sum()/refserie.shape[0]
        if debug>=2:
            print('remove_duplicated_columns:%i:%.2f:%.2f'%(i,ratio,ratio2))
        # YOu can have cases where half of the rows are empty hence ratio = 0.5 and it is not at all the same content
        if ratio>threshold:
            delidx+=[i]
        elif ratio2>threshold:
            delidx+=[i]
        else:
            keepidx+=[i]
    if len(delidx)>0:
        return df_pandas.iloc[:,list(set(keepidx))]
    return df_pandas

def fast_filter_ownership_table(df_pandas,debug=0,penalty=1):
    """
    First step are some quick filters to nail down only the ownership tables
    The idea is that you should never find these words in an ownership table...
    """
    if df_pandas is None:
        return -199
    if df_pandas.shape[0]==0:
        return -199
    df_pandas_str = df_pandas.astype(str).sum().astype(str).sum()
    df_pandas_str = df_pandas_str.lower()
    # Minimum character replacement
    df_pandas_str = re.sub(r'[^\x00-\x7F]+', ' ', df_pandas_str)
    res=0
    remove_one_point= [
        'accounting','proxy statement','on table','your vote counts','proxyvote','vote online','vote','check box',
        'agenda','since','proposal','meeting','www.','salary','incentive','award','retirement','vesting',
        'termination','diversity','restricted','audit','audit matters','gaap','leadership','governance framework',
        'risk','weighted average','percentile','proposal','information','return','p/e','growth','income','earning',
        'base salary','part','item','staff','disclosure','disclose','factor','propert','supplement','registrant',
        'accountant',' fee','signature','summary','management','exhibit','corportate','governance',' age ','technology',
        'compensation','committee','discussion',' profit','consolidated','pre-tax','rsu','psa grant',
        'dcf','payout','target','2022','2023','2021','period','performance','issuance','future','weighted','exercise',
        'warrants','options','rights','equity','approve',
        'stock options','exercise price','grant','nasdaq'
        ]
    remove_five_point=['dollar value']
    add_five_point= ['beneficial owner','beneficially owned','number of shares',
                     ' as a group','persons)','individuals)','percent of shares outstanding']
    add_one_point  = ['blackrock','vanguard','percent','all director',
                      'nominees for director','common stock ','outstanding','shares']
    for expression in remove_one_point:
        if expression in df_pandas_str:
            #if debug>=3:
            #    print('Negative : '+expression)
            res-=1*penalty
    for expression in remove_five_point:
        if expression in df_pandas_str:
            #if debug>=3:
            #    print('Negative 5: '+expression)
            res-=5*penalty
    for expression in add_one_point:
        if expression in df_pandas_str:
            #if debug >= 3:
            #    print('Positive : '+expression)
            res+=1
    for expression in add_five_point:
        if expression in df_pandas_str:
            #if debug >= 3:
            #    print('Positive 5: '+expression)
            res+=8
    return res

