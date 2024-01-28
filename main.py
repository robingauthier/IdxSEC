import pandas as pd
import numpy as np
from pprint import pprint
from .edgar_utils import convert_to_sec_url,get_random_form
from lxml import etree
from . import g_edgar_folder

pd.options.display.max_columns = 10000
pd.set_option('display.expand_frame_repr', False)

def train_models():
    from .extract_table_content_model import train_tree
    train_tree()
    from .extract_table_content_section_model import train_model_v1
    train_model_v1()

def main_loc(fname,debug=1):
    resd = {}
    log=''
    url=convert_to_sec_url(fname)
    resd['url'] = url
    print('\'fname\':\''+fname)
    print(url)
    log+='fname is:\n'
    log+=fname
    log += 'url is:\n'
    log+=url

    with open(g_edgar_folder + fname, 'rt') as f:
        html = f.read()

    from .extract_table_content_v1 import extract_table_content
    searchlogdf, content_table = extract_table_content(fname, debug=0)
    if content_table is not None:
        from .extract_table_content_model import predict_table_content
        searchlogdf['pred0']=predict_table_content(searchlogdf)

        from .extract_table_content_section_model import predict_table_content_section
        content_table['pred']=predict_table_content_section(content_table,debug=0)


        content_table=content_table.merge(searchlogdf[['hash','pred0']],on='hash',how='left')
        log+=str(content_table[['section','pred0','pred']])
        if debug>0:
            print(content_table[['section','pred0','pred']])

        # we need to get the next href
        content_table_u=content_table.groupby('href').agg({'section': 'sum', 'pred': 'max'}).reset_index()
        content_table_u['npred'] = content_table_u['pred'].shift(1)

        href_ownership_section = None
        href_ownership_section_post = None
        if content_table['pred'].sum()>0:
            href_ownership_section=content_table.loc[lambda x:x['pred']>0]['href'].iloc[0]
            href_ownership_section_post=content_table_u.loc[lambda x:x['npred']>0]['href'].iloc[-1]


        resd['href_ownership_section']=href_ownership_section
        resd['href_ownership_section_post'] = href_ownership_section_post

        pprint(resd)

        from .extract_structure_form_v1 import extract_structure_form
        structd=extract_structure_form(fname)

        # checking the attributes that can hide href references
        struct_names=pd.Series({k: v['name'] for k, v in structd.items()}).dropna()
        struct_id = pd.Series({k: v['id'] for k, v in structd.items()}).dropna()
        struct_df=pd.concat([
            struct_names.to_frame('href').assign(attrib='name').reset_index(),
            struct_id.to_frame('href').assign(attrib='id').reset_index()],axis=0,sort=False)

        start_id = struct_df.loc[lambda x:x['href']==href_ownership_section[1:]]['index'].iloc[0]
        end_id = struct_df.loc[lambda x: x['href'] == href_ownership_section_post[1:]]['index'].iloc[0]
        resd['start_id']=start_id
        resd['end_id'] = end_id

        from .extract_structure_form_v2 import filter_html
        nhtml = filter_html(fname,fromid=start_id-1,toid=end_id,debug=0)

        nfname=g_edgar_folder+'../data_supervised/html_ownership_section.html'
        with open(nfname, 'w') as outfile:
            outfile.write(nhtml)
        print('Pls read '+nfname)

        html=nhtml

    # List of tables
    from .convert_html_table_dataframe import convert_html_table_dataframe
    restree = etree.HTML(html)
    tables=restree.xpath('//table')
    ltables=[]
    for table in tables:
        table_html=etree.tostring(table,method='html',encoding=str)
        dfloc=convert_html_table_dataframe(table_html)
        ltables+=[dfloc]
    print('List of table shapes')
    print([x.shape for x in ltables])
    # only tables with multiple columns are elegible

    # and now I need a text version of that
    from .convert_html_text import iter_html_to_text,convert_dict_to_text
    restree = etree.HTML(nhtml)
    _, ltags = iter_html_to_text(restree, elemi=0, lres=[], debug=3)
    if len(ltags) == 0:
        return ''
    txt = ''.join([convert_dict_to_text(x) for x in ltags])
    nfname=g_edgar_folder+'../data_supervised/html_ownership_section.txt'
    with open(nfname, 'w') as outfile:
        outfile.write(txt)


    # maintenant faut produire un text file avec la bonne section
    import pdb;pdb.set_trace()




# ipython -i -m IdxSEC.main
if __name__=='__main__':
    #fname='form14A_cik1626971_asof20230428_0001558370-23-007172.txt'
    #train_models()
    fname='form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'  # issue on this one!!!
    #fname=get_random_form()
    #main_loc(fname=fname)
    test_1()