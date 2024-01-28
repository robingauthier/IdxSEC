import requests
import pandas as pd
import datetime
import numpy as np
import os
import shutil
import time
from . import g_edgar_folder
from .filesys import UnZipFile


# syntax for looking at last files is :
# ls -lhtr | tail -n 20

def get_sec_master_file(iurl='2023/QTR2',redownload=False):
    """full url looks like :
    https://www.sec.gov/Archives/edgar/full-index/2017/QTR3/master.zip
    It contains the address of all the filling submitted

    it returns a dataframe if it succeeds
    otherwise it returns None
    """
    filename = 'edgar_form_list.pkl'
    tfname0 = os.path.join(*[g_edgar_folder, filename])

    url = 'https://www.sec.gov/Archives/edgar/full-index/%s/master.zip'%iurl
    headers = {
        'User-Agent': 'My User Agent 1.0',
        'From': 'youremail@domain.example'  # This is another valid field
    }
    if (not os.path.exists(tfname0)) or redownload:
        f = requests.get(url,headers=headers)
        if f.status_code!=200:
            print('Issue in downloading the master file')
            print(f.content)
            if 'Access Denied' in f.content:
                return None
            raise(ValueError('Issue downloading the master file'))
        with open(g_edgar_folder+'temp_sec.zip', 'wb') as ff:
            ff.write(f.content)
        tmpfname=g_edgar_folder+'temp_sec.zip'
        nfile = UnZipFile(tmpfname)
        tfname=os.path.join(*[nfile, 'master.idx'])

        nowstr=pd.to_datetime('now',utc=True).strftime('%Y%m%d_%H%M')
        shutil.copy(tfname,g_edgar_folder+'secmaster_%s_%s.csv'%(iurl.replace('/',''),nowstr))

        df = pd.read_csv(os.path.join(*[nfile, 'master.idx']), sep='|', skiprows=11, encoding="ISO-8859-1")
        df.columns = 'CIK|Company Name|Form Type|Date Filed|Filename'.split('|')
        df['cik'] = df['CIK'].astype(str).apply(lambda x: x if len(x) == 10 else '0' * (10 - len(x)) + x)
        df.to_pickle(tfname0)
        os.remove(tmpfname)
    else:
        print('Using saved file')
    df = pd.read_pickle(tfname0)
    return df

def get_sec_form(formtype,dateform,cik,url):
    """
    fname is :form14A_cik1195737_asof20230630_0001193125-23-180532.txt
    url looks like :
    'https://www.sec.gov/Archives/edgar/data/1195737/0001193125-23-180532.txt
    'https://www.sec.gov/Archives/edgar/data/CIK/accession_number.txt'
    ---> ceci est un fichier texte
    pour avoir le folder il faut   :
    https://www.sec.gov/Archives/edgar/data/1195737/000119312523180532  ( donc tout attache)
    ensuite dans ce cas le html est :
    https://www.sec.gov/Archives/edgar/data/1195737/000119312523180532/0001193125-23-180532-index.html
    puis :
    https://www.sec.gov/Archives/edgar/data/1195737/000119312523180532/d483405ddef14a.htm
    """
    headers = {
        'User-Agent': 'My User Agent 1.0',
        'From': 'youremail@domain.example'  # This is another valid field
    }
    formtype2=formtype.replace(' ','').replace('DEF','')
    dateform2=pd.to_datetime(dateform).strftime('%Y%m%d')
    filename='form'+formtype2+'_'+'cik'+str(cik)+'_'+'asof'+dateform2+'_'+os.path.basename(url)
    tfname=os.path.join(*[g_edgar_folder,filename])
    if os.path.exists(tfname):
        #print('Already saved %s' % tfname)
        return None
    f = requests.get(url,headers=headers)
    print('Saving %s'%tfname)
    with open(tfname,'wt') as ff:
        ff.write(f.content.decode('utf-8'))
    time.sleep(3)
    return tfname

def get_iurl(today=None,offset=0):
    if today is None:
        today = datetime.datetime.today()
    if offset!=0:
        today=today+datetime.timedelta(days=offset * 90)
    year = today.year
    quarter = (today.month - 1) // 3 + 1
    quarter_str = f'QTR{quarter}'
    return f'{year}/{quarter_str}'

def main(formtypes=['DEF 14A','DEFA14A'],iurl=None,iurloffset=0,ids=None,redownload=False,fromdt=None):

    #redownload=False

    if iurl is None:
        iurl=get_iurl(offset=iurloffset)
    df = get_sec_master_file(iurl=iurl,redownload=redownload)
    df = df[df['Form Type'].isin(formtypes)]
    if ids is not None:
        df = df[df['CIK'].isin(ids)]
    if df.shape[0]==0:
        raise(ValueError('you filtered every forms'))
    df['Date Filed']=pd.to_datetime(df['Date Filed'],format='%Y-%m-%d')
    df = df.sort_values('Date Filed',ascending=False)
    print('Last forms mentioned in master file : ')
    print( df.head(20))
    if fromdt is not None:
        df=df[df['Date Filed']>fromdt]
        print('Last forms after fromdt : %s'%fromdt)
        print(df.head(20))
    #import pdb;pdb.set_trace()
    if df.shape[0]==0:
        return
    print('Making sure we downloaded them all')
    df['url'] = 'https://www.sec.gov/Archives/' + df['Filename']

    lfnames=[] # list of new file names created
    for id,row in df.iterrows():
        fnameloc=get_sec_form(row['Form Type'],row['Date Filed'],row['CIK'],row['url'])
        if fnameloc is not None:
            lfnames+=[fnameloc]
        if len(lfnames)>=50:
            break
    print('TODO: change len(lfnames) max')
    return lfnames

def test_1():
    resp=get_iurl(today=datetime.datetime(2023, 1, 1))
    assert resp=='2023/QTR1','issue'
    resp=get_iurl(today=datetime.datetime(2023, 6, 1))
    assert resp=='2023/QTR2','issue'
    resp=get_iurl(today=datetime.datetime(2023, 8, 15))
    assert resp=='2023/QTR3','issue'
    resp=get_iurl(today=datetime.datetime(2023, 12, 15))
    assert resp=='2023/QTR4','issue'
    resp = get_iurl(today=datetime.datetime(2023, 12, 15),offset=1)
    assert resp == '2024/QTR1', 'issue'

# ipython -i -m IdxSEC.edgar_dwnd_v1
if __name__=='__main__':
    #df=get_sec_master_file()
    #main(ids=[320193,1018724]) # cik of apple
    #from .mapCIK import download_map_cik_ticker
    #mdf=download_map_cik_ticker()
    #id=mdf[mdf['ticker'] == 'msft']['cik'].iloc[0]
    main(fromdt=pd.to_datetime('2024-01-18'))



