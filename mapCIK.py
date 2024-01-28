import requests
import os
import pandas as pd
import numpy as np
from .filesys import find_last_file
from . import g_edgar_folder
ref_path='D:\\data\\cikmap.csv'
def download_map_cik_ticker(redownload=True):
    url = 'https://www.sec.gov/include/ticker.txt'
    headers = {
        'User-Agent': 'My User Agent 1.0',
        'From': 'youremail@domain.example'  # This is another valid field
    }
    fnow=pd.to_datetime('now',utc=True).strftime('%Y%m%d')
    tfname=g_edgar_folder+'map_cik_%s.pkl'%fnow

    if not os.path.exists(tfname):
        print('downloading the map cik')
        f = requests.get(url, headers=headers)
        if f.status_code!=200:
            print('Issue in downloading the master file')
            print(f.content)
            raise(ValueError('Issue downloading the master file'))
        fcontent=f.content
        with open(g_edgar_folder+'temp_map_cik.txt', 'wb') as ff:
            ff.write(fcontent)
        df=pd.read_table(g_edgar_folder+'temp_map_cik.txt',header=None)
        df.columns=['ticker','cik']
        df.to_pickle(tfname)
    return pd.read_pickle(tfname)



# ipython -i -m IdxSEC.mapCIK
if __name__=='__main__':
    df=download_map_cik_ticker()

