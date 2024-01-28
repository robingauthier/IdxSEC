import pandas as pd
import numpy as np
import re
from .convert_html_table_dataframe import convert_html_table_dataframe
from . import g_edgar_folder
import matplotlib
import matplotlib.pyplot as plt

# we first collect information and compute a table
# that classifies if we have noice,a table or a footnote
# and only once we have done this we go to the block
# and extract the footnote


def re_search_all(regex,text,gname):
    matches = re.finditer(regex, text)
    lr=[]
    for match in matches:
        lr+=[{'match':match.group(gname),
              'startpos':match.start(),
              'endpos':match.end()}]
    return pd.DataFrame(lr)

def collect_footnotes_in_table_old(table_loc):
    """from a table """
    # this must be a sub function
    lfootnotes = []
    for jj in range(table_loc.shape[1]):
        col_txt = ' '.join(table_loc.iloc[:, jj].dropna().astype(str).tolist())
        col_mdf = re_search_all('\((?P<id>\d+)\)', col_txt, gname='id')
        if col_mdf.shape[0]== 0:
            continue
        col_mdf['coli']=jj
        lfootnotes+=[col_mdf]
    rdf=pd.concat(lfootnotes,axis=0,sort=False)
    return rdf

def collect_footnotes_in_table(table_loc):
    """from a table """
    # this must be a sub function
    lfootnotes = []
    for jj in range(table_loc.shape[1]):
        for ii in range(table_loc.shape[0]):
            if pd.isna(table_loc.iloc[ii, jj]):
                continue
            col_txt = str(table_loc.iloc[ii, jj])
            col_mdf = re_search_all('\((?P<id>\d+)\)', col_txt, gname='id')
            if col_mdf.shape[0]== 0:
                continue
            col_mdf['coli']=jj
            col_mdf['rowi'] = ii
            col_mdf['maxpos']=len(col_txt)
            #import pdb;pdb.set_trace()
            lfootnotes+=[col_mdf]
    if len(lfootnotes)==0:
        return pd.DataFrame()
    rdf=pd.concat(lfootnotes,axis=0,sort=False)
    return rdf


def collect_footnotes_from_struct(lstruct):
    """
    The job is to just pull the information
    the classification task will be done later on

           match  startpos  endpos    tag  elemi  nbfootnote  table_nbcol  table_nbrow  colstartpos  colendpos  coli  rowi
    42     1       721     724  table   5020           3          2.0          5.0         36.0       39.0   1.0   2.0
    43     2       928     931  table   5020           3          2.0          5.0        243.0      246.0   1.0   2.0
    44     3      1071    1074  table   5020           3          2.0          5.0        386.0      389.0   1.0   2.0
    45     2       140     143  table   5338           9          8.0         24.0         11.0       14.0   0.0   5.0
    46     3       185     188  table   5338           9          8.0         24.0         13.0       16.0   0.0   7.0
    47     4       337     340  table   5338           9          8.0         24.0         13.0       16.0   0.0  13.0
    48     5       395     398  table   5338           9          8.0         24.0         16.0       19.0   0.0  15.0
    49     6       440     443  table   5338           9          8.0         24.0         14.0       17.0   0.0  17.0
    """
    lfootnotes=[]

    for elem in lstruct:
        txtloc=elem['content']

        mdf=re_search_all('\((?P<id>\d+)\)', txtloc,gname='id')
        if mdf.shape[0]==0:
            # you can be between 2 footnotes and not have any pattern
            mdf=pd.DataFrame({'match':[np.nan],'startpos':[np.nan],
                              'endpos':[np.nan],'elemi':[elem['elemi']],
                              'blockid':[elem['blockid']],
                              'tag':[elem['tag']]})
            lfootnotes += [mdf]
            continue

        mdf['tag']=elem['tag']
        mdf['elemi']=elem['elemi']
        if 'blockid' in elem.keys():
            mdf['blockid'] = elem['blockid']
        mdf['nbfootnote']=mdf.shape[0]

        if elem['tag']=='table' and txtloc.startswith('{') and txtloc.endswith('}'):
            table_loc=pd.read_json(txtloc)
            mdf['table_nbcol'] = table_loc.shape[1]
            mdf['table_nbrow'] = table_loc.shape[0]
            table_mdf=collect_footnotes_in_table(table_loc)
            if table_mdf.shape[0]>0:
                table_mdf=table_mdf.rename(columns={'startpos':'colstartpos','endpos':'colendpos','maxpos':'colmaxpos'})
                mdf=mdf.merge(table_mdf,on=['match'],how='left')
                # when match is not unique we introduce duplicates
                mdf=mdf.drop_duplicates(subset=['match','coli','rowi','colstartpos','colendpos'])

        lfootnotes+=[mdf]
    rdf=pd.concat(lfootnotes,axis=0,sort=False).reset_index(drop=True)
    return rdf

def model_categorize_footnotes(rdf):
    """
    The text contains pollution like :
    (10) votes per share
    g shares held: (1) directly in your name as the stockholder of record, and (2) for you as the beneficial owner in street

    """
    rdf['match_num'] = pd.to_numeric(rdf['match'],errors='coerce')
    rdf=rdf[rdf['match_num'].fillna(10)<=30]

    lr=[]
    elemis=rdf['elemi'].unique().tolist()
    for elemi in elemis:
        resloc={'elemi':elemi,
                'score_footnote':0,
                'score_table':0,
                'score_noise':0,
                }
        rdfloc=rdf.loc[lambda x:x['elemi']==elemi].copy()
        resloc['blockid']=rdfloc['blockid'].iloc[0]

        # if the row number is correlated to the id it has more chances to be a footnote
        if 'rowi' in rdfloc.columns:
            corr= rdfloc[['match', 'rowi']].astype(np.float64).corr().iloc[0,1]
        else:
            corr=np.nan
        # to know if we are in a table or not
        pct_table=(~pd.isna(rdfloc['colstartpos'])).sum()/rdfloc.shape[0]
        # when the id is at the start of the text it is more likely to be a footnote
        pct_startcol=(rdfloc['colstartpos']<=5).sum()/rdfloc.shape[0] if pct_table>0.5 else np.nan
        pct_endcol = (rdfloc['colendpos'] >=rdfloc['colmaxpos']-5).sum() / rdfloc.shape[0] if pct_table > 0.5 else np.nan

        # if a lot of text is in the column 0
        pct_col0=(rdfloc['coli']==0).sum()/rdfloc.shape[0] if pct_table>0.5 else np.nan
        # when the id is at the start of the text it is more likely to be a footnote
        pct_col0_startcol = ((rdfloc['coli'] == 0)*(rdfloc['colstartpos'] <= 5)).sum() / rdfloc.shape[0] if pct_table>0.5 else np.nan
        # when it is a text the start pos
        pct_startpos = (rdfloc['startpos'] <= 5).sum() / rdfloc.shape[0]

        resloc['pct_col0']=pct_col0
        resloc['corr'] = corr
        resloc['pct_table']=pct_table
        resloc['pct_startcol'] = pct_startcol
        resloc['pct_endcol']=pct_endcol
        resloc['pct_col0_startcol'] = pct_col0_startcol
        resloc['table_nbcol']=rdfloc['table_nbcol'].iloc[0]
        resloc['max_id_repeat']=rdfloc['match'].value_counts().max()
        resloc['pct_startpos']=pct_startpos

        if pct_col0_startcol>0.7:
            resloc['score_footnote']+=5
        #if pct_startcol<0.3:
        #    resloc['score_table']+=1
        if pct_endcol>0.7:
            resloc['score_table'] += 1
        if resloc['table_nbcol']<=2:
            resloc['score_footnote'] += 1
        if resloc['table_nbcol']>=3:
            resloc['score_table'] += 1
        # the fact that there are some repeats is a sign of table
        if resloc['max_id_repeat']>1:
            resloc['score_table'] += 1

        # dans le cas d'un text et non d'une table
        if pct_startpos>0.8:
            resloc['score_footnote'] += 1
        else:
            resloc['score_noise'] += 1
        if rdfloc['startpos'].min()>100:
            resloc['score_noise'] += 1
        #if not rdfloc['match'].astype(int).is_monotonic_increasing:
        #    resloc['score_noise'] += 1
        if rdfloc.shape[0]>5:
            resloc['score_noise'] -= 5
        if pct_col0>0.7:
            resloc['score_noise'] -= 5
        #if elemi==1643:
        #    import pdb;pdb.set_trace()
        if False:
            print('-'*10)
            print(rdfloc)
            print(resloc)
        # si le matchid ne monte pas faut enlever c'est absurde
        # when the table has only 2 columns it is more likely to be a footnote
        lr+=[resloc]
        #import pdb;pdb.set_trace()
    if False:
        rdf['match'].plot()
        plt.show()
    rdf=pd.DataFrame(lr)
    #print(rdf.loc[lambda x: x['elemi'] == 1643])
    # adding a point if a footnote follows closely a table
    rdf['tag']='table' if pct_table>0 else 'div'
    rdf['istable']=1*(rdf['score_table']>=1)*(rdf['score_table']>rdf['score_footnote'])*(rdf['score_table']>rdf['score_noise'])
    rdf['ptableid']=np.where(rdf['istable']>0,rdf['blockid'],np.nan)
    rdf['ptableid'] =rdf['ptableid'].ffill()
    rdf['cond']=1*(rdf['blockid']<=rdf['ptableid']+20)*(rdf['score_footnote']>0)
    rdf['pscore_footnote'] =rdf['score_footnote']
    rdf['score_footnote']=np.where((rdf['istable']==0)&(rdf['cond']>0),rdf['score_footnote']+1,rdf['score_footnote'])
    #print(rdf.loc[lambda x:x['elemi']<=8466].tail(40)[['elemi','pscore_footnote','score_footnote','score_noise','istable']])
    #import pdb
    #pdb.set_trace()
    #print(rdf.loc[lambda x:x['elemi']==1643])
    #import pdb;pdb.set_trace()
    rdf = rdf.drop(['pscore_footnote','cond','ptableid','istable'],axis=1)

    # adding a point if a text is in between 2 footnotes
    # it means the footnote is split on multiple divs
    win=3
    rdf['score_footnote_p']=rdf['score_footnote'].shift(1).rolling(window=win).max().fillna(0)
    rdf['score_footnote_n'] = rdf['score_footnote'].shift(-win).rolling(window=win).max().fillna(0)
    rdf['blockid_p']=rdf['blockid'].shift(1).rolling(window=win).max().fillna(rdf['blockid'])
    rdf['blockid_n'] = rdf['blockid'].shift(-win).rolling(window=win).min().fillna(rdf['blockid'])
    rdf['cond']=1*(rdf['score_footnote_p']>0)*(rdf['score_footnote_n']>0)*(rdf['tag']!='table')
    rdf['cond2'] = 1*(rdf['blockid_n']<=rdf['blockid_p']+4)
    rdf['newval'] = 0.5*(rdf['score_footnote_n']+rdf['score_footnote_p'])
    rdf['cond3']=rdf['cond2']*rdf['cond']*(rdf['score_footnote']<rdf['newval'])
    # 6396 is the issue for instance
    rdf['score_footnote']=np.where(rdf['cond3']>0,rdf['newval'],rdf['score_footnote'])
    #print(rdf[['elemi', 'blockid', 'blockid_p', 'blockid_n', 'score_footnote', 'cond', 'newval', 'score_table', 'tag',
    #           'score_footnote_p', 'score_footnote_n']].loc[lambda x: x['elemi'] >= 1643 - 20].head(30))
    #import pdb;pdb.set_trace()
    rdf=rdf.drop(['cond','cond2','cond3','score_footnote_p','score_footnote_n','blockid_p','blockid_n'],axis=1)


    rdf['score_sum']=rdf['score_table']+rdf['score_footnote']
    rdf['score_noise']=np.where(rdf['score_sum']==0,rdf['score_noise']+1,rdf['score_noise'])
    #rdf = rdf.loc[lambda x:x['score_sum']>0].copy()
    #rdf['pred']=np.where(rdf['score_table']>=rdf['score_footnote'],'table','footnote')
    rdf['pred']=rdf[['score_footnote','score_noise','score_table']].idxmax(axis=1).str.replace('score_','')
    rdf['max']=rdf[['score_footnote', 'score_noise', 'score_table']].max(axis=1)
    rdf['pred'] = np.where(rdf['score_footnote'] >= rdf['max'], 'footnote', rdf['pred'])
    rdf['pred'] = np.where(rdf['score_table']>=rdf['max'],'table',rdf['pred'])

    rdf['tableid']=np.where(rdf['pred']=='table',rdf['elemi'],np.nan)
    rdf['tableid'] =rdf['tableid'].ffill()
    rdf['tableid'] = rdf['tableid'].fillna(0)
    return rdf


def concat_footnotes(rdf,lstruct):
    # collect content of footnotes
    rdf = rdf.drop_duplicates(subset=['elemi'])
    rdf = rdf.loc[lambda x: x['pred'] == 'footnote']

    structd = {v['elemi']: v for v in lstruct}
    tableids = rdf['tableid'].unique()
    lr = []
    #import pdb;pdb.set_trace()
    for tableid in tableids:
        rdfloc = rdf.loc[lambda x: x['tableid'] == tableid]
        #rdfloc = rdfloc.loc[lambda x: x['pred'] == 'footnote']
        if rdfloc.shape[0] == 0:
            continue
        for id, row in rdfloc.iterrows():
            elemi=row['elemi']
            elem = structd[row['elemi']]
            txtloc = elem['content']
            is_table=(elem['tag'] == 'table') and txtloc.startswith('{') and txtloc.endswith('}')
            if not is_table:
                mdf = re_search_all('\((?P<id>\d+)\)', txtloc, gname='id')
                resloc={
                        'txt':txtloc,
                        'tableid':row['tableid'],
                        'elemi':elemi,
                        'tag':'table' if is_table else 'div'}
                if mdf.shape[0]==0:
                    resloc['match'] = np.nan
                else:
                    mdf=mdf.iloc[[0]] # we only keep the first match
                    resloc['match']=mdf['match'].iloc[0]
                lr+=[resloc]
            if is_table:
                table_loc = pd.read_json(txtloc)
                ntable_loc=table_loc.astype(str).apply(lambda row: ' '.join(row), axis=1)
                for nid,nrow in ntable_loc.items():
                    txtloc=nrow
                    nmdf = re_search_all('\((?P<id>\d+)\)', txtloc, gname='id')
                    if nmdf.shape[0]==0:
                        continue
                    #if nmdf.shape[0]==1:
                    nmdf=nmdf.iloc[[0]] # we only keep the first match
                    resloc = {'match': nmdf['match'].iloc[0],
                              'txt': txtloc,
                              'tableid': row['tableid'],
                              'elemi': elemi,
                              'tag':'table' if is_table else 'div'
                              }
                    lr += [resloc]
                    #import pdb;pdb.set_trace()
    rdf=pd.DataFrame(lr)
    rdf['match']=rdf['match'].ffill()

    return rdf


def compute_score_document(lstruct):
    """the more we find rows starting by (\d) better"""
    rl=[]
    for elem in lstruct:
        txtloc=str(elem['content'])
        mdf = re_search_all('\((?P<id>.{1,5})\)', txtloc, gname='id')
        rl+=[mdf]
    rdf=pd.concat(rl,axis=0,sort=False)
    score=(rdf['startpos']<=10).sum()
    return score

def compute_len_document(lstruct):
    """the more we find rows starting by (\d) better"""
    rl=[]
    for elem in lstruct:
        txtloc=str(elem['content'])
        rl+=[{'elemi':elem['elemi'],'len':len(txtloc)}]
    rdf=pd.DataFrame(rl)
    #import pdb;pdb.set_trace()
    #score=(rdf['startpos']<=10).sum()/rdf.shape[0]
    rdf['len']=np.where(rdf['len']>5000,5000,rdf['len'])
    return rdf['len'].mean()

def find_best_win(fname):
    # form14A_cik1551901_asof20230425_0001104659-23-049676.txt
    # <re.Match object; span=(0, 56), match='form14A_cik1551901_asof20230425_0001104659-23-049>
    # https://www.sec.gov/Archives/edgar/data/1551901/000110465923049676
    #    win     score  score_width  score_len
    # 0    1  0.215517   320.550136        369
    # 1    3  0.215517   320.552846        369
    # 2   10  0.181034   414.740351        285
    # 3   20  0.172414   590.920000        200
    # 4   40  0.120690  1114.150943        106
    # 5   70  0.077586  1816.476923         65
    # 6  100  0.077586  1935.524590         61
    # 7  200  0.077586  1967.766667         60
    # 8  400  0.068966  2135.454545         55
    from .blocks_html_v1 import blocks_html
    lr=[]
    for win in [1,3,10,20,40,70,100,200,400]:
        lstruct, _ = blocks_html(fname,win=win)
        score_pattern_start=compute_score_document(lstruct)
        score_width=compute_len_document(lstruct)
        score_len = len(lstruct)
        # score : how many rows start with a letter that is not uppercase
        ratio=score_pattern_start/score_len
        lr+=[{'win':win,'score_pattern':score_pattern_start,
              'score_width':score_width,'score_len':score_len,'score':ratio}]
    rdf=pd.DataFrame(lr)
    print(rdf)
    #import pdb;pdb.set_trace()
    return rdf
def collect_footnotes(fname):
    from .convert_html_text import convert_dict_to_text
    from .blocks_html_v1 import blocks_html

    win=find_best_win(fname)
    lstruct, _ = blocks_html(fname, win=10)
    #import pdb;pdb.set_trace()

    # adding block id
    blockid=0
    for elem in range(len(lstruct)):
        blockid+=1
        lstruct[elem]['blockid']=blockid

    # Same for the block version
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname=g_edgar_folder + '../data_supervised/html_block.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    rdf1=collect_footnotes_from_struct(lstruct)
    #print(rdf)
    rdf=model_categorize_footnotes(rdf1)
    nice_rdf=rdf[['elemi','blockid','tableid','pred','tag','score_noise','score_footnote','score_table']]
    #print(nice_rdf)
    # ft stands for footnote
    ft0=concat_footnotes(nice_rdf, lstruct)
    # last step is to group them by match and tableid
    ft=ft0.groupby(['tableid', 'match']).agg({'txt': 'sum', 'elemi': 'min'}).reset_index()\
    .sort_values(['tableid','elemi'])
    return {'ft':ft,'ft_step1':ft0,'ft_class':nice_rdf,'step2':rdf,'step1':rdf1}

def test_1():
    mydata=[
        {'fname':'form14A_cik1652044_asof20230421_0001308179-23-000736.txt',
         'lcat':[
             {'elemi':14623,'cat':'footnote','id':None},
             {'elemi':5338,'cat':'table','id':'ownership'},
             {'elemi':5861,'cat':'footnote','id':'ownership'},
             {'elemi':6377,'cat':'table','id':'director_compensation'},
             {'elemi': 6534, 'cat': 'footnote','id':'director_compensation'},
         ]},
        {'fname':'form14A_cik1787306_asof20230418_0001787306-23-000033.txt',
         'lcat': [
             {'elemi': 8886, 'cat': 'table', 'id': 'ownership'},
             {'elemi':9342,'cat':'footnote','id':'ownership'},
             {'elemi': 9345, 'cat': 'footnote', 'id': 'ownership'},
             {'elemi': 9372, 'cat': 'footnote', 'id': 'ownership'},#9405
             {'elemi': 9405, 'cat': 'footnote', 'id': 'ownership'},
         ]
         },
        {'fname':'form14A_cik1583708_asof20230517_0001583708-23-000028.txt',
         'lcat':[
         {'elemi': 6113, 'cat': 'table', 'tableid': 6113,'note':'ownership'},
         {'elemi':6583,'cat':'footnote','tableid':6113,'note':'ownership'},
         {'elemi': 6949, 'cat': 'footnote', 'tableid': 6113, 'note': 'ownership'},
         {'elemi': 5361, 'cat': 'table','tableid':5361,'note':'payvsperformance'},
         {'elemi': 5552, 'cat': 'footnote', 'tableid': 5361, 'note': 'payvsperformance'},# le text a ete coupe
         {'elemi': 5559, 'cat': 'footnote', 'tableid': 5361, 'note': 'payvsperformance'}
         ]},
        {'fname':'form14A_cik1837240_asof20240119_0000950170-24-005813.txt',
         'lcat':[
             {'elemi':5088, 'cat': 'table', 'tableid': 5088,'note':'ownership'},
             {'elemi': 6384, 'cat': 'footnote', 'tableid': 5088, 'note': 'ownership'},
             {'elemi': 6400, 'cat': 'footnote', 'tableid': 5088, 'note': 'ownership'},
             {'elemi': 6461, 'cat': 'footnote', 'tableid': 5088, 'note': 'ownership'},
         ]},
        {'fname':'form14A_cik1615219_asof20230501_0001615219-23-000034.txt',
         'lcat':[
             {'elemi': 3111, 'cat': 'table', 'tableid': 3111, 'note': 'ownership'},
             {'elemi': 3231, 'cat': 'footnote', 'tableid': 3111, 'note': 'ownership'},
             {'elemi': 3255, 'cat': 'footnote', 'tableid': 3111, 'note': 'ownership'},
         ]},
        {'fname':'form14A_cik1744494_asof20230425_0001829126-23-002903.txt',
         'lcat': [
             {'elemi': 2366, 'cat': 'table', 'tableid': 2366, 'note': 'ownership'},
             {'elemi': 2579, 'cat': 'footnote', 'tableid': 2366, 'note': 'ownership'},
             {'elemi': 2623, 'cat': 'footnote', 'tableid': 2366, 'note': 'ownership'},
         ]},


        {'fname': 'form14A_cik1453593_asof20230608_0001493152-23-020490.txt',
         'lcat': [
             {'elemi': 8475, 'cat': 'table', 'tableid': 8475, 'note': 'ownership1'},
             {'elemi': 8599, 'cat': 'footnote', 'tableid': 8475, 'note': 'ownership1'},
             {'elemi': 8649, 'cat': 'footnote', 'tableid': 8475, 'note': 'ownership1'},
             {'elemi': 8701, 'cat': 'table', 'tableid': 8701, 'note': 'ownership2'},
             {'elemi': 8949, 'cat': 'footnote', 'tableid': 8701, 'note': 'ownership2'},
             {'elemi': 9143, 'cat': 'footnote', 'tableid': 8701, 'note': 'ownership2'},
             {'elemi': 9151, 'cat': 'footnote', 'tableid': 8701, 'note': 'ownership2'},
         ]},

        {'fname': 'form14A_cik1702510_asof20230428_0001702510-23-000028.txt',
         'lcat': [
             {'elemi': 510, 'cat': 'table', 'tableid': 510, 'note': 'ownership1'},
             {'elemi': 770, 'cat': 'footnote', 'tableid': 510, 'note': 'ownership1'},
         ]},

        {'fname': 'form14A_cik1655759_asof20230428_0001655759-23-000034.txt',
         'lcat': [
             {'elemi': 7607, 'cat': 'table', 'tableid': 7607, 'note': 'ownership1'},
             {'elemi': 7932, 'cat': 'footnote', 'tableid': 7607, 'note': 'ownership1'},
             {'elemi': 7998, 'cat': 'footnote', 'tableid': 7607, 'note': 'ownership1'},
         ]},

    ]

# ipython -i -m IdxSEC.collect_footnotes_ownership
if __name__=='__main__':
    #fname='form14A_cik1850787_asof20230426_0001213900-23-032897.txt'# ok
    #fname = 'form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    from .edgar_utils import get_random_form
    fname=get_random_form()
    #fname='form14A_cik1787306_asof20230418_0001787306-23-000033.txt'
    #fname='form14A_cik1652044_asof20230421_0001308179-23-000736.txt'

    #fname='form14A_cik1551901_asof20230425_0001104659-23-049676.txt' # super hard this one
    # as the divs are coming one after each other (1) then this is ...
    #fname='form14A_cik1583708_asof20230517_0001583708-23-000028.txt'
    #fname='form14A_cik1837240_asof20240119_0000950170-24-005813.txt'
    #fname='form14A_cik1453593_asof20230608_0001493152-23-020490.txt'
    #fname='form14A_cik1702510_asof20230428_0001702510-23-000028.txt'
    #fname='form14A_cik1655759_asof20230428_0001655759-23-000034.txt'
    print(fname)
    from .edgar_utils import convert_to_sec_url
    print(convert_to_sec_url(fname))
    rd=collect_footnotes(fname)
    #print(rd['res'])
    if True:
        ft=rd['ft']
        ftc=ft.copy()
        ftc['txt']=ftc['txt'].str[:50]
        print(ftc)
    #import pdb;pdb.set_trace()

    #tableid=8886
    #print(rd['step2'].loc[lambda x:x['elemi']>=500].drop(['newval','corr','max_id_repeat'],axis=1))

