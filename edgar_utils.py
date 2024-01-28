import re
import requests
import pandas as pd
import numpy as np
import os
import shutil
import time
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 100

def convert_to_sec_url(filename):
    """
    fname is :form14A_cik1195737_asof20230630_0001193125-23-180532.txt
    pour avoir le folder il faut   :
    https://www.sec.gov/Archives/edgar/data/1195737/000119312523180532  ( donc tout attache)

    """
    # Define the pattern for extracting CIK and accession number
    pattern = r'form(?P<ftype>\w+)_cik(?P<cik>\d+)_asof(?P<asof>\d+)_(?P<n1>\d+)-(?P<n2>\d+)-(?P<n3>\d+)\.txt'

    # Search for the pattern in the filename
    match = re.match(pattern, filename)
    print(match)

    if match is not None:
        #import pdb;pdb.set_trace()
        # Extract CIK and accession number from the matched groups
        cik = match.group('cik')
        n1 = match.group('n1')
        n2 = match.group('n2')
        n3 = match.group('n3')
        # Construct the SEC URL
        sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{n1}{n2}{n3}"
        print(sec_url)
        return sec_url
    else:
        # Return None if the pattern is not found
        return np.nan

def get_forms():
    from .filesys import find_files_using_glob
    from . import g_edgar_folder
    print(g_edgar_folder+'form14A_cik*.txt')
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=list(set(lfilesdf['file'].tolist()))
    return lfiles

def get_random_form(not_in=[]):
    import random
    lfiles=get_forms()
    if len(not_in)>0:
        lfiles=[x for x in lfiles if not x in not_in]
    fname= random.choice(lfiles)
    print(fname)
    return fname

def read_form(fname):
    from . import g_edgar_folder
    url = convert_to_sec_url(fname)

    print('fname: ' + fname)
    print(url)

    with open(g_edgar_folder + fname, 'rt') as f:
        html = f.read()
    return html

# ipython -i -m IdxSEC.edgar_utils
if __name__=='__main__':
    # Example usage:
    filename = "form14A_cik1195737_asof20230630_0001193125-23-180532.txt"
    sec_url = convert_to_sec_url(filename)
    print(sec_url)
    assert sec_url=='https://www.sec.gov/Archives/edgar/data/1195737/000119312523180532'