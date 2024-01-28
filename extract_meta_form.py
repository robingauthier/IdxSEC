

import re
import time
import os
import copy
from pprint import pprint
import pandas as pd
import numpy as np
from lxml import etree
from pprint import pprint
from glob import glob
import random



def extract_metainfo_from_form(fname,debug=0):
    resd = {}
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    meta_txt = etree.tostring(restree.xpath('//sec-header')[0],method='text',encoding=str)
    txt = etree.tostring(restree, method='text', encoding=str)
    resd['asof'] = re.search("FILED\sAS\sOF\sDATE:(?P<a>.*)", meta_txt).group('a').replace('\t','')
    resd['ftype'] = re.search("CONFORMED\sSUBMISSION\sTYPE:(?P<a>.*)", meta_txt).group('a').replace('\t','')
    resd['name'] = re.search("COMPANY\sCONFORMED\sNAME:(?P<a>.*)", meta_txt).group('a').replace('\t','')
    resd['cik'] = re.search("CENTRAL\sINDEX\sKEY:(?P<a>.*)", meta_txt).group('a').replace('\t','')
    resd['nbelem'] = len(restree.xpath('//*'))
    resd['asof'] = pd.to_datetime(resd['asof'],format='%Y%m%d')
    # trying to detect the voting forms
    search_vote_1 = re.search('You cannot use this notice to vote your shares',txt)
    search_vote_2 = re.search('For a convenient way to view proxy materials and VOTE', txt)
    search_vote_3 = re.search('This is not a ballot', txt)
    resd['isballot']= ((search_vote_1 is not None) or
                       (search_vote_2 is not None) or
                       (search_vote_3 is not None)
                       )
    resd['fsize']=os.path.getsize(g_edgar_folder+fname)/1024
    search_ownership_1=re.search('SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',txt)
    resd['hasOwnership']= ((search_ownership_1 is not None))
    return resd

# ipython -i -m IdxSEC.extract_meta_form
if __name__=='__main__':
    from . import g_edgar_folder
    from lxml import etree
    fname='form14A_cik1195737_asof20230630_0001193125-23-180532.txt'
    resd = extract_metainfo_from_form(fname)
    print(resd)


