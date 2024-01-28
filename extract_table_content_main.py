import os.path

import pandas as pd
import numpy as np
from pprint import pprint
from .edgar_utils import convert_to_sec_url,get_random_form
from lxml import etree
from . import g_edgar_folder

pd.options.display.max_columns = 10000
pd.set_option('display.expand_frame_repr', False)

def infer_elemid_ownership_section_using_content_table(fname,debug=0):
    """
    This is the result of multiple scripts :
    1 - Collect the possible content table
    2 - Run model to infer the ownership section in the content table
    3 - Find the href of this ownership section
    4- Find the next href which will be the end of the ownership section

    5- map this back to elemId so that we know the id number of start/end of the section
    then we report in a dictionary
    resd['start_id']=start_id
    resd['end_id'] = end_id
    """
    resd={'fname':fname}

    from .extract_table_content_v1 import extract_table_content
    searchlogdf, content_table = extract_table_content(fname, debug=debug)
    if debug:
        print(searchlogdf)
        print(content_table)

    if content_table is None:
        resd['href_ownership_section'] = None
        resd['href_ownership_section_post'] = None
        resd['start_id'] = None
        resd['end_id'] = None
        return resd

    #from .extract_table_content_model import predict_table_content
    #searchlogdf['pred0']=predict_table_content(searchlogdf)
    searchlogdf['pred0'] = 1.0

    #from .extract_table_content_section_model import predict_table_content_section
    from .extract_table_content_section_model_bench import predict_table_content_section
    content_table['pred']=predict_table_content_section(content_table,debug=debug)

    content_table=content_table.merge(searchlogdf[['hash','pred0']],on='hash',how='left')
    if debug>0:
        print(content_table[['section','pred0','pred']])

    # we need to get the next href. u means unique here
    content_table_u=content_table.groupby('href').agg({'section': 'sum', 'pred': 'max','row':'mean'}).reset_index()

    # Keeping initial order
    content_table_u=content_table_u.sort_values('row',ascending=True)
    content_table = content_table.sort_values('row', ascending=True)

    # The next section after ownership is :
    content_table_u['npred'] = content_table_u['pred'].shift(1)

    #import pdb;pdb.set_trace()
    resd['content_table'] = content_table
    resd['content_table_u']=content_table_u

    href_ownership_section = None
    href_ownership_section_post = None
    if content_table['pred'].sum()>0:
        href_ownership_section=content_table.loc[lambda x:x['pred']>0]['href'].iloc[0]
    if content_table_u['npred'].sum()>0:
        # we have the case ownership is the last section ...
        href_ownership_section_post=content_table_u.loc[lambda x:x['npred']>0]['href'].iloc[-1]
    else:
        href_ownership_section_post=None

    resd['href_ownership_section']=href_ownership_section
    resd['href_ownership_section_post'] = href_ownership_section_post
    if href_ownership_section is None:
        resd['href_ownership_section'] = None
        resd['href_ownership_section_post'] = None
        resd['start_id'] = None
        resd['end_id'] = None
        return resd
    from .extract_structure_form_v1 import extract_structure_form
    structd=extract_structure_form(fname,as_dict=True)

    # checking the attributes that can hide href references
    struct_names=pd.Series({k: v['name'] for k, v in structd.items()}).dropna()
    struct_id = pd.Series({k: v['id'] for k, v in structd.items()}).dropna()
    struct_df=pd.concat([
        struct_names.to_frame('href').assign(attrib='name').reset_index(),
        struct_id.to_frame('href').assign(attrib='id').reset_index()],axis=0,sort=False)

    start_id = struct_df.loc[lambda x:x['href']==href_ownership_section[1:]]['index'].iloc[0]
    if href_ownership_section_post is not None:
        end_id = struct_df.loc[lambda x: x['href'] == href_ownership_section_post[1:]]['index'].iloc[0]
    else:
        end_id = None
    resd['attrib_df']=struct_df
    resd['start_id']=start_id
    resd['end_id'] = end_id
    if end_id is not None:
        if start_id>=end_id:
            print('issue in infer_elemid_ownership_section_using_content_table')
    return resd


def test_1():
    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'
    resd=infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section']=='#T4','issue'
    assert resd['href_ownership_section_post'] in ['#T9934','#T3'],'issue' # right answer is T9934.

    fname = 'form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section'] == '#i56e0089e07bc44eeb9fc7e4ee35b4343_31', 'issue'
    assert resd['href_ownership_section_post'] =='#i56e0089e07bc44eeb9fc7e4ee35b4343_34', 'issue'

    fname='form14A_cik1795815_asof20230613_0001795815-23-000006.txt'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    # on this one the title of the section is not part of the a tag
    assert resd['href_ownership_section'] is None, 'issue'
    assert resd['href_ownership_section_post']  is None, 'issue'


    fname='form14A_cik1455365_asof20230424_0001104659-23-049042.txt'
    #'https://www.sec.gov/Archives/edgar/data/1455365/000110465923049042'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section'] == '#tSOOC', 'issue'
    assert resd['href_ownership_section_post'] =='#tPTBV', 'issue'

    fname='form14A_cik1838513_asof20230605_0001193125-23-160816.txt'
    #'https://www.sec.gov/Archives/edgar/data/1838513/000119312523160816'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section']=='#tx500320_11','issue'
    assert resd['href_ownership_section_post'] == '#tx500320_12', 'issue'

    fname='form14A_cik102037_asof20230623_0000102037-23-000039.txt'
    #'https://www.sec.gov/Archives/edgar/data/102037/000010203723000039'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section'] is None, 'issue'
    assert resd['href_ownership_section_post'] is None, 'issue'

    # this one is a tricky one because content table is split on 2 columns
    # hence the _post will be wrong
    fname='form14A_cik1759824_asof20230427_0001193125-23-122351.txt'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section'] == '#toc445706_23', 'issue'


    fname='form14A_cik1868778_asof20230426_0001628280-23-013255.txt'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section']=='#i4158db56eb504dc1b4ec41695a7f7dbc_124','issue'
    assert resd['href_ownership_section_post'] == '#i4158db56eb504dc1b4ec41695a7f7dbc_127', 'issue'

    # here we had an issue on row id due to the 2 tables of ownership
    fname = 'form14A_cik1844862_asof20230410_0001104659-23-043334.txt'
    resd = infer_elemid_ownership_section_using_content_table(fname)
    assert resd['href_ownership_section']=='#tCRAR','issue'
    assert resd['href_ownership_section_post'] == '#tSOOC', 'issue'

    #fname = get_random_form()
    #url=convert_to_sec_url(fname)
    #print(url)
    #resd = infer_elemid_ownership_section_using_content_table(fname)
    #import pdb;pdb.set_trace()

def test_2():
    ### tests cannot be that harsh
    mydata=[
        {'fname':'form14A_cik1844862_asof20230410_0001104659-23-043334.txt',
         'href_ownership_section':'#tCRAR',
         'href_ownership_section_post':'#tSOOC'
         },

    ]

def extract_table_content(fname,debug=0):
    """a nice utility on top to write a json file
    was used for Airflow
    """
    resd=infer_elemid_ownership_section_using_content_table(fname,debug=debug)
    from .edgar_utils import convert_to_sec_url
    resd['url']=convert_to_sec_url(fname)

    from . import g_ndata_folder
    nfname=os.path.join(*[g_ndata_folder,'content_table_'+os.path.basename(fname)])

    # dataframe is not serializable
    if 'content_table_u' in resd.keys():
        resd['content_table_u']=resd['content_table_u'][['section','href','pred']].to_dict()
    if 'content_table' in resd.keys():
        resd.pop('content_table')
    if 'attrib_df' in resd.keys():
        resd.pop('attrib_df')

    import json
    from .json_utils import NpEncoder
    with open(nfname,'wt') as f:
        json.dump(resd,f,indent=4,cls=NpEncoder)
    return resd


# ipython -i -m IdxSEC.extract_table_content_main
if __name__=='__main__':
    #fname='form14A_cik1626971_asof20230428_0001558370-23-007172.txt'
    #train_models()
    #fname='form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    #fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'  # issue on this one!!!
    #fname=get_random_form()
    #main_loc(fname=fname)
    #test_1()
    fname = 'form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    fname='form14A_cik1844862_asof20230410_0001104659-23-043334.txt'
    fname='form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    fname='form14A_cik1770787_asof20230428_0001770787-23-000021.txt'
    #extract_table_content(fname,debug=1)
    resd = infer_elemid_ownership_section_using_content_table(fname, debug=1)
    print( resd['content_table'][['section','row','href']])