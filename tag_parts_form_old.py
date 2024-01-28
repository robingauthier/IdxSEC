
import tabula
import re
import time
import os
import copy
from pprint import pprint
import pandas as pd
import numpy as np
from lxml import etree
import sys
from pprint import pprint
from glob import glob
import random
from .fast_filter_onwership_table import fast_filter_ownership_table,filter_dimension_dataframe
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200
g_tmpfolder = 'D:\\edgar\\'
lref = [
    'SIGNIFICANT SHAREHOLDERS',
    'Stock Owned by Directors',
    'Ownership of Voting Securities',
    'Beneficial owners of 5%',
    'Ownership of Voting Securities by Certain Beneficial Owners',
    'Principal Shareholders',
    'CERTAIN BENEFICIAL OWNERS',
    'Greater than 5% Security Holders',
    'Five Percent Holders',
    'Principal Stockholders',
    'Ownership by Directors',
    'Principal Stockholders and Ownership by Directors and Executive Officers',
    'Stock Ownership',
    'SHIP OF PRINCIPAL STOCKHOLDERS', # span is splitting
    'SECURITY OWNERSHIP',
    'MANAGEMENT OWNERSHIP',
    'SECURITY OWNERSHIP OF PRINCIPAL STOCKHOLDERS',
    'Beneficial Ownership of More Than 5%',
    'beneficial owner of more than 5%',
    'Security Ownership of Certain Beneficial Owners',
    'Executive Officer and Director Stock Ownership',
    'Director and executive officer ownership',
    'Beneficial Ownership',
    'Beneficial Ownership of Directors',
    'Beneficial Ownership of Shares',
    'INFORMATION ON STOCK OWNERSHIP',
    'STOCK OWNERSHIP INFORMATION',
    'OWNERSHIP OF CERTAIN BENEFICIAL OWNERS',
    'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
    'SECURITY OWNERSHIP OF MANAGEMENT',
    'Principal Stockholders and Beneficial Owners',
    'Beneficial Owners',
    'Security Ownership',
    'Principal Holders of Stock',
    'Holders of Stock',
    #'OWNERSHIP',
    'Ownership of Equity Securities of the Company',
]
lother = [
    'equity incentive',
    'Stockholder Approval',
    'Awards Granted',
    'Plan Benefits',
    'RELATIONSHIPS AND RELATED PARTY TRANSACTIONS',
    'Compensation Discussion & Analysis',
    'Proxy Summary',
    'Agenda Item',
    # 'Annual Report', too many
    'Compensation Tables',
    'COMPENSATION DISCUSSION AND ANALYSIS',
    'Role of the Compensation Committee',
    'Summary Compensation Table',
    'Outstanding Equity Awards',
    'Option Exercises and Stock Vested',
    'Director Compensation',
    'Outstanding Equity Awards',
    'AUDIT MATTERS',
    'Audit Committee Report',
    'Board and Committee Matters',
    'Additional Voting Matters',
    'CEO Pay Ratio', 'Code of Ethics', 'Corporate Governance Guidelines''Pension Benefits',
    'ADDITIONAL INFORMATION', 'EQUITY COMPENSATION', 'Stock Ownership Guidelines',
    'director stock ownership requirement', 'Report of the HRC Committee', 'Executive Compensation Table',
    'Related-Party Transactions', 'Annual Meeting Information', 'Shareholder Proposals ', 'Stock Ownership Policy'
]
def find_in_text(txtloc,elem,elemi,lres):
    txtlocl = txtloc.lower()
    isref = False
    iselem = False
    word = None
    for reft in lref:
        if reft.lower() in txtlocl:
            isref = True
            word = ('*' * 20) + reft + ('*' * 20)
    for reft in lother:
        if reft.lower() in txtlocl:
            iselem = True
            word = reft
    if isref and not iselem:
        lres += [{'kind': 'ref', 'elemi': elemi, 'word': word, 'tag': elem.tag}]
    if iselem and not isref:
        lres += [{'kind': 'elem', 'elemi': elemi, 'word': word, 'tag': elem.tag}]
    return lres
def fast_filter_explanation_section(txtloc):
    nbstarts = re.findall('\(\d+\)',txtloc)
    score = len(nbstarts)
    return score
def clean_txt(txtloc):
    txtloc=txtloc.replace('\n','').replace('\t','')
    #if 'Lawrence J. Ellison (2)' in txtloc:
    #    import pdb
    #    pdb.set_trace()
    return txtloc
def find_explanation_section(txtloc,elem,elemi,lres,df=None):
    #elemloc=etree
    # sometimes the explanation section is a table where the first column has (1), (2)...
    #if elem.tag=='table':
    #    print(df)
    #    import pdb
    #    pdb.set_trace()
    # sometimes the explanation has sup
    if len(elem.xpath('/sup'))>0:
        import pdb
        pdb.set_trace()
    explanation_score = fast_filter_explanation_section(txtloc)
    if explanation_score>0:
        lres += [{'kind': 'exp', 'elemi': elemi, 'word': clean_txt(txtloc[:200]), 'tag': elem.tag}]
    return lres
def iter_elems3(elem,elemi=0,lres=[],debug=1):
    #if (elemi==162) and (elem.tag=='table'):
    #    import pdb
    #    pdb.set_trace()
    if len(lres)>=2000:
        import pdb
        pdb.set_trace()
    if debug>=2:
        print(elemi)
        print(len(lres))
        print('-'*5)
        #txtloc=etree.tostring(elem, method='text', encoding=str).lower()
        #print(txtloc[:500])
        #print('-'*10)
    #if ('security' in debugtxt) and ('ownership' in debugtxt) and len(debugtxt)<5000:
    #    import pdb
    #    pdb.set_trace()
    elemi+=1
    if elem.tag=='table' :
        txtloc = etree.tostring(elem, method='html', encoding=str)
        txtloc2 = etree.tostring(elem, method='text', encoding=str)
        if len(txtloc)>100:
            try:
                df_pandas = pd.read_html(txtloc)[0]
            except Exception as e:
                df_pandas=None
            df_pandas1 = filter_dimension_dataframe(df_pandas, max=1)
            if df_pandas1 is None:
                return elemi,lres
            fast_filter_score = fast_filter_ownership_table(df_pandas1, debug=debug,penalty=0.5)
            lres=find_explanation_section(txtloc2, elem, elemi, lres,df_pandas1)
            if fast_filter_score>0:
                if debug>=2:
                    print(df_pandas1)
                lres += [{'kind': 'table',
                          'elemi': elemi,
                          'score': fast_filter_score,
                          'elem':elem,
                          'col1':'|'.join(df_pandas.iloc[:,0].astype(str).tolist())}]
            else:
                lres += [{'kind': 'table_info', 'elemi': elemi, 'score': -100,'word':clean_txt(txtloc2[:200])}]
        else:
            lres += [{'kind': 'table_info', 'elemi': elemi, 'score': -100,'word':clean_txt(txtloc2[:200])}]
        return elemi,lres
    nb_childs = len(elem)
    txtloc = etree.tostring(elem, method='text', encoding=str)
    # For ORACLE 14A this is the text :
    # '\n SECURITY\xa0OWNERSHIP\xa0OF\xa0CERTAIN\xa0BENEFICIAL\xa0OWNERS\xa0\nAND MANAGEMENT\n\xa0\xa0\n\xa0\n31\n\xa0\n'
    txtloc = re.sub(r'[^\x00-\x7F]+', ' ', txtloc)  # removes all unicode characters
    if (len(txtloc)<100) and (nb_childs!=0):
        lres=find_in_text(txtloc,elem,elemi,lres)
        lres=find_explanation_section(txtloc,elem,elemi,lres)
        return elemi, lres
    if (nb_childs==0):
        lres = find_in_text(txtloc,elem,elemi,lres)
        lres = find_explanation_section(txtloc, elem, elemi, lres)
        return elemi, lres
    for child in elem:
        if debug>=2:
            print('Digging into childs : ')
        elemi,lres = iter_elems3(child,elemi=elemi,lres=lres,debug=debug)
    return elemi,lres

def infer_interesting_tables(df):
    df['imp']=np.nan
    df['imp'] = np.where(df['kind']=='ref',1,df['imp'])
    df['imp'] = np.where(df['kind'] == 'elem', 0, df['imp'])
    df['imp']=df['imp'].ffill()
    df['imp2']=1*(df['kind']=='ref')
    df['imp3'] =0
    win = int(max(0.04*df.shape[0],10))
    for i in range(win):
        df['imp3']+=df['imp2'].shift(i)
    return df[(df['kind']=='table')&((df['imp']==1)|(df['imp3']>0))]

def infer_interesting_exp(tagdf):
    tagdf['imp']=np.nan
    tagdf['imp'] = np.where(tagdf['kind'] == 'ref', 1, tagdf['imp'])
    tagdf['imp'] = np.where(tagdf['kind'] == 'elem', 0, tagdf['imp'])
    tagdf['imp']=tagdf['imp'].ffill()
    tagdf['imp2']= 1 * (tagdf['kind'] == 'ref')
    tagdf['imp3'] =0
    win = int(max(0.04 * tagdf.shape[0], 10))
    for i in range(win):
        tagdf['imp3']+=tagdf['imp2'].shift(i)
    return tagdf[(tagdf['kind'].isin(['table_exp', 'exp'])) & ((tagdf['imp'] == 1) | (tagdf['imp3'] > 0))]

def mainloc(restree,debug=0):
    elem_tables = restree.xpath('//table')
    _, ltags = iter_elems3(restree, elemi=0, lres=[], debug=debug)
    if len(ltags) == 0:
        return []
    tagdf = pd.DataFrame(ltags)
    nb_tables = tagdf[tagdf['kind'].isin(['table', 'table_info'])].shape[0]
    if nb_tables > len(elem_tables):
        print_orange('issue we have more tables than what is in the doc')
        print_orange('From iterelem :%i  vs xpath %i' % (nb_tables, len(elem_tables)))
        import pdb
        pdb.set_trace()
    # First condition is to be below a ref element
    tagdf['imp']=np.nan
    tagdf['imp'] = np.where(tagdf['kind'] == 'ref', 1, tagdf['imp'])
    tagdf['imp'] = np.where(tagdf['kind'] == 'elem', 0, tagdf['imp'])
    tagdf['imp']=tagdf['imp'].ffill()
    tagdf['imp2']= 1 * (tagdf['kind'] == 'ref')
    tagdf['imp3'] =0
    win = int(max(0.04 * tagdf.shape[0], 10))
    for i in range(win):
        tagdf['imp3']+=tagdf['imp2'].shift(i)
    tagdf['imp3'] = np.sign(tagdf['imp3'])
    tagdf['cond1']=tagdf['imp3']
    # Second condition is to find a table with positive score
    tagdf['cond2']  = tagdf['cond1']*(tagdf['kind']=='table')*(tagdf['score']>5)
    tagdf['cond2_id'] = tagdf['cond2'].cumsum()
    tagdf['cond2_id'] = tagdf['cond2_id'].fillna(0.0)
    # Third condition it to have some explanations after the table
    lres=[]
    for cond2_id in tagdf['cond2_id'].unique().tolist():
        if cond2_id==0:
            continue
        tagdfloc  = tagdf[(tagdf['cond2_id']==cond2_id)&(tagdf['cond1']>0)].copy()
        if tagdfloc.shape[0]==0:
            continue
        if tagdfloc[tagdfloc['kind']=='exp'].shape[0]==0:
            print('No explanations found below the table..')
            continue
        lres+=[tagdfloc[(tagdfloc['kind']=='table')&(tagdfloc['score']>5)]]
        lres += [tagdfloc[(tagdfloc['kind'] == 'exp')]]
    rdf = pd.concat(lres,axis=0,sort=False)
    tagdf.to_csv('D:\\data\\temp0.csv')
    rdf.to_csv('D:\\data\\temp.csv')
    import pdb
    pdb.set_trace()
    #tagdf['cond3'] =

    #return tagdf[(tagdf['kind'].isin(['table_exp', 'exp'])) & ((tagdf['imp'] == 1) | (tagdf['imp3'] > 0))]

    #table_elems = tagdf2['elem'].tolist()
def main():
    from .extract_meta_form import extract_metainfo_from_form
    lfiles = glob(g_tmpfolder + '*.txt')
    #random.shuffle(lfiles)
    ii = 0
    for fname in lfiles:
        fname = os.path.basename(fname)
        link = ('https://www.sec.gov/Archives/edgar/data/' +
                fname.split('_')[0] + '/' +
                fname.split('_')[1].replace('.txt', '').replace('-', ''))
        with open(g_tmpfolder+fname,'rt') as f:
            ff = f.read()
        restree = etree.HTML(ff)
        resd={}
        resd=extract_metainfo_from_form(restree,resd)
        if resd['ftype']!= 'DEF 14A':
            continue
        print('-' * 10)
        print(fname)
        print(link)
        pprint(resd)
        #elemi, lres = iter_elems(restree)
        elemi, lres = iter_elems3(restree)
        rdf = pd.DataFrame(lres)
        rdf2 =infer_interesting_tables(rdf)
        print(rdf2)
        import pdb
        pdb.set_trace()
        #iter_elems(restree)
        continue

        if rdf.shape[0]<=10:
            continue
        print(rdf)
        found_ref = (rdf[rdf['kind']=='ref'].shape[0]>0)
        found_score = rdf[rdf['score']>=0].shape[0]
        if found_score>=2:
            import pdb
            pdb.set_trace()
        if not found_ref:
            print(rdf)
            import pdb
            pdb.set_trace()
        #import pdb
        #pdb.set_trace()

# ipython -i -m FloatEdgar_v3.tag_parts_form
if __name__=='__main__':
    fname='315189_0001558370-23-000200.txt' # ok
    fname='1390777_0001193125-22-059759.txt' # needs merging with content table
    #fname='896878_0001104659-22-121633.txt' # needs merging with content table
    fname='1063761_0001104659-23-036085.txt' # hard one. I think needs mergin with content table
    #fname='59558_0001193125-23-100016.txt' # ok fine
    #fname='896878_0001104659-22-121633.txt' # hard one. needs merging with content table
    fname='1000228_0001193125-22-096767.txt'
    fname='868857_0001104659-23-007991.txt' #  incroyable il te sort des tables qui n'existent pas !!!
    fname='56873_0001104659-22-063488.txt'
    fname = '910329_0001628280-23-014346.txt'
    fname='1744489_0001193125-22-012592.txt'
    fname='1915657_0001193125-23-093991.txt'
    fname = '4447_0001193125-22-105599.txt'
    fname='1164863_0001171200-23-000143.txt'
    fname = '24545_0001308179-23-000559.txt'
    fname='1742924_0001308179-23-000191.txt'
    fname='1418819_0001418819-23-000014.txt'
    fname = '1341439_0001193125-22-250158.txt'
    fname='1000228_0001193125-22-096767.txt'
    fname = '1915657_0001193125-23-093991.txt'
    #main()
    if True:
        nfname = fname
        link='https://www.sec.gov/Archives/edgar/data/'+nfname.split('_')[0]+'/'+nfname.split('_')[1].replace('.txt','').replace('-','')
        print(link)
        with open(g_tmpfolder+fname,'rt') as f:
            ff = f.read()
        restree = etree.HTML(ff)
        mainloc(restree)
        #ctdf=extract_content_table_from_form(restree)
        #elemi,lres = iter_elems3(restree,debug=0)
        #rdf= pd.DataFrame(lres)
        #rdf.to_csv('D:\\data\\temp.csv',index=False)

