import os.path
import shutil
from io import StringIO
import numpy as np
import pandas as pd
import re
from pprint import pprint
from lxml import etree
from . import g_edgar_folder
from . import myhash
from .extract_table_content_scoring import lpos,lneg
from .text_utils import clean_txt
from .table_utils import remove_duplicated_columns
import distance #pip install Distance

# if the table of content is a table and we can compute some scores then we extract it.
# this version is for an ML algorithm

def score_text_list(txtloc,lpos,lneg):
    txtlocl = txtloc.lower()
    scorep=0
    scoren = 0
    for wpos in lpos:
        scorep+=txtlocl.count(wpos)
    for wneg in lneg:
        scoren+=txtlocl.count(wneg)
    return {'scorep':scorep,'scoren':scoren}

def extra_table_content_html(restree,starti=0):
    lr=[]
    i=starti
    #phref=''
    for elem in restree.xpath('//tr'):
        i+=1
        htmlloc = etree.tostring(elem, method='html', encoding=str)
        restreeloc = etree.HTML(htmlloc)
        j=0
        for elemloc in restreeloc.xpath('//a[@href]'):
            j+=1
            section = etree.tostring(elemloc, method='text', encoding=str)
            section=str(clean_txt(section))
            href=elemloc.xpath('@href')[0]
            # this logic was introduced because of form14A_cik1844862_asof20230410_0001104659-23-043334.txt
            #
            #if (j!=1) and (phref!=href):
            #    i+=1
            lr+=[{'section':section,'href':href,'row':i}]
            #phref = href
    rdf=pd.DataFrame(lr)
    return rdf,i

def extract_table_content_loc(restree,meta_fname='',debug=0):
    """
    we iterate on every table we output 2 things:
    - a dictionary that will be feed to a ML that contains features
    - the actual table
    """
    lres=[]
    lresdf=[]
    pos=0
    poslarge=0
    starti=0
    for elem in restree.xpath('//table'):
        resd = {}
        pos+=1
        htmlloc = etree.tostring(elem, method='html', encoding=str)
        txtloc = etree.tostring(elem, method='text', encoding=str)
        try:
            l_df_pandas = pd.read_html(StringIO(htmlloc))
        except Exception as e:
            # todo : pandas.errors.ParserError
            continue
        if len(l_df_pandas)==0:
            continue

        # Now we can work on creating features
        if l_df_pandas[0].shape[0]>=5:
            poslarge +=1
        restreeloc = etree.HTML(htmlloc)
        resd['nhref'] = len(restreeloc.xpath('//a[@href]'))
        if resd['nhref']<=0:
            continue
        resd['pos']=pos
        resd['poslarge'] = poslarge

        # this hash is the unique id, same in the structure
        resd['hash']=myhash(etree.tostring(elem, method='html', encoding=str))
        resd['nrows'] = l_df_pandas[0].shape[0]
        resd['ncols'] = l_df_pandas[0].shape[0]

        scored=score_text_list(txtloc,lpos,lneg)
        resd['score_txt_p']=scored['scorep']
        resd['score_txt_n'] = scored['scoren']
        resd['has_text1']=1*('content' in txtloc.lower())
        resd['has_text2'] = 1 * ('ownership' in txtloc.lower())
        resd['has_text3'] = 1 * ('audit' in txtloc.lower())
        resd['has_text4'] = 1 * ('javascript' in txtloc.lower())
        resd['fname'] = meta_fname
        #resd['ctext'] =

        # we don't use l_df_pandas we prefer to see the href
        # if we do not use starti the n tables can have same row id
        dfloc,starti=extra_table_content_html(restreeloc,starti=starti)
        dfloc['hash']=resd['hash']
        dfloc['fname']=meta_fname
        #import pdb;pdb.set_trace()
        if debug>0:
            print(resd)
            print(dfloc)
            print('-'*20)

        lresdf+=[dfloc]
        lres+=[resd]
    if len(lresdf)==0:
        return None,None
    resdf = pd.concat(lresdf, axis=0, sort=False).reset_index(drop=True)
    logdf=pd.DataFrame(lres)
    logdf['ntables']=logdf.shape[0]
    return logdf,resdf

def extract_table_content(fname,debug=0):
    """wrapper to call"""
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    return extract_table_content_loc(restree,meta_fname=fname,debug=debug)

def test_run(fname = 'form14A_cik1816017_asof20230501_0000950170-23-016022.txt'):
    from . import g_edgar_folder

    nfname=g_edgar_folder + '../data_supervised/table_content.txt'
    nfnamelog = g_edgar_folder + '../data_supervised/table_content_log.txt'

    searchlogdf,content_table = extract_table_content(fname)

    with open(nfname, 'wt') as f:
        if content_table is not None:
            content_table.to_string(f)
        else:
            f.write('No content table found')

    with open(nfnamelog, 'wt') as f:
        searchlogdf.to_string(f)

def main(reset=False):
    """loops on every form and concatenates """
    from .edgar_utils import get_forms
    from .filesys import append_to_file
    lfiles=get_forms()
    nfname = g_edgar_folder + '../data_supervised/table_content_all.txt'
    nfnamelog = g_edgar_folder + '../data_supervised/table_content_log_all.txt'
    if reset:
        try:
            os.remove(nfname)
            os.remove(nfnamelog)
        except Exception as e:
            pass

    i=0
    for fname in lfiles:
        i+=1
        searchlogdf, content_table = extract_table_content(fname)
        if content_table is None:
            continue

        append_to_file(content_table,nfname)
        append_to_file(searchlogdf, nfnamelog)
        if i%50==0:
            print('Step i=%i'%i)

# ipython -i -m IdxSEC.extract_table_content_v1
if __name__=='__main__':
    #test_run()
    #main(reset=True)
    nfnamelog = g_edgar_folder + '../data_supervised/table_content_log_all.txt'
    df=pd.read_csv(nfnamelog)
    tdf=df.groupby('fname').agg({'hash':'nunique'}).sort_values('hash')
    print(tdf[tdf['hash']==10])