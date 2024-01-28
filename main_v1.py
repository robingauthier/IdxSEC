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

    with open(g_edgar_folder + fname, 'rt') as f:
        html = f.read()

    from .extract_structure_form_v1 import extract_structure_form
    from .convert_html_text import convert_dict_to_text
    lstruct_full = extract_structure_form(fname)

    from .blocks_html_v1 import blocks_html
    lstruct, _ = blocks_html(fname)
    from .page_footer_clean import page_footer_clean_from_struct
    #todo
    # Saving down
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct_full])
    nfname=g_edgar_folder + '../data_supervised/html_full.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    # Same for the block version
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname=g_edgar_folder + '../data_supervised/html_block.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    ### Step 1 is infering where the ownership section is

    from .extract_table_content_main import infer_elemid_ownership_section_using_content_table
    resd1=infer_elemid_ownership_section_using_content_table(fname)

    from .classify_ownership_text_hmm_v1 import classify_ownership_text_from_struct
    resd2 = classify_ownership_text_from_struct(lstruct)

    # comparing the 2 informations
    from .classify_ownership_text_hmm_v1 import overlap_score
    score=overlap_score(resd1['start_id'],resd1['end_id'],resd2['infer_start_id'],resd2['infer_end_id'],maxlen=len(lstruct_full))

    if score>0.6:
        resd['start_id']=resd1['start_id']
        resd['end_id'] = resd1['end_id'] if resd1['end_id'] is not None else resd1['infer_end_id']
    else:
        resd['start_id']=resd2['infer_start_id']-20
        resd['end_id'] = resd2['infer_end_id']+20

    # End of the step 1 is to save a new html only with the ownership section
    from .extract_structure_form_v2 import filter_html
    nhtml = filter_html(fname,fromid=resd['start_id']-1,toid=resd['end_id'],debug=0)

    nfname=g_edgar_folder+'../data_supervised/html_ownership_section.html'
    with open(nfname, 'w') as outfile:
        outfile.write(nhtml)
    print('Pls read '+nfname)

    print('-'*20)
    print('-' * 20)
    ### Step 2 now : eeee
    from .collect_footnotes_ownership import collect_footnotes
    rd = collect_footnotes(fname)
    #rd['ft'] has the footnotes

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


    # TODO:# ipython -i -m IdxSEC.classify_rows_ownership_table_supervised


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




# ipython -i -m IdxSEC.main_v1
if __name__=='__main__':
    #fname='form14A_cik1626971_asof20230428_0001558370-23-007172.txt'
    #train_models()
    fname='form14A_cik1524025_asof20230417_0001628280-23-011813.txt'
    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'  # issue on this one!!!
    fname=get_random_form()
    main_loc(fname=fname)
    #test_1()