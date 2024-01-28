import re
import copy
import pandas as pd
import numpy as np
from distance import jaccard
from .hierarchical_clustering import hierarchical_clustering_wthreshold

# some forms repeat a lot of info
def preprocess(lstruct):
    nstruct=[]
    for elem0 in lstruct:
        elem=copy.deepcopy(elem0)
        txt0=elem['content']
        elem['content']=re.sub(r'\d+', '', elem['content'])
        elem['content'] = re.sub(r'\s+', '', elem['content'])
        nstruct += [elem]
    return nstruct

def filter_short_elems(lstruct,maxlen=50):
    nstruct=[]
    for elem in lstruct:
        if len(elem['content'])<=3:
            continue
        if len(elem['content'])>maxlen:
            continue
        nstruct += [elem]
    return nstruct

def page_footer_clean_from_struct(lstruct0,debug=False):
    lstruct=preprocess(lstruct0)
    nstruct=filter_short_elems(lstruct,maxlen=400)
    nstructd={v['elemi']:v for v in nstruct}
    n = len(nstruct)
    idx = [elem['elemi'] for elem in nstruct]
    mat=np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            assert i!=j,'issue'
            mat[i,j]=jaccard(nstruct[i]['content'],nstruct[j]['content'])
            mat[j, i] =mat[i,j]
    matdf=pd.DataFrame(mat)
    matdf.index=idx
    matdf.columns = idx
    clusdf=hierarchical_clustering_wthreshold(matdf, threshold=0.40, debug=False)
    clusdf=clusdf.rename(columns={'col':'elemi'})
    clusdf = clusdf.loc[lambda x:x['clustercnt']>=15]
    if debug:
        for clusterid in clusdf['clusterid'].unique().tolist():
            clusdfloc = clusdf[clusdf['clusterid'] == clusterid]
            print('-'*10)
            for id,row in clusdfloc.iterrows():
                elemi=row['elemi']
                print(nstructd[elemi])

    del_elems = clusdf['elemi'].tolist()
    rstruct = [x for x in lstruct0 if not x['elemi'] in del_elems]
    return {'lstruct':rstruct,'del_elems':del_elems,'matdf':matdf}

def page_footer_clean(fname,debug=True):
    from .extract_structure_form_v1 import extract_structure_form
    from .convert_html_text import convert_dict_to_text
    from .blocks_html_v1 import blocks_html
    from . import g_edgar_folder

    lstruct0, _ = blocks_html(fname)
    rd=page_footer_clean_from_struct(lstruct0,debug=debug)
    lstruct=rd['lstruct']
    del_elems=rd['del_elems']

    # Saving down
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct0])
    nfname = g_edgar_folder + '../data_supervised/html_block.txt'
    print('Open text file ' + nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname = g_edgar_folder + '../data_supervised/html_block_nofootpage.txt'
    print('Open text file ' + nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)
    return rd

def test_1():
    from .extract_structure_form_v1 import extract_structure_form
    from .convert_html_text import convert_dict_to_text
    from .blocks_html_v1 import blocks_html
    from . import g_edgar_folder


    fname='form14A_cik29534_asof20230411_0001104659-23-044057.txt'
    lstruct0, _ = blocks_html(fname)
    rd=page_footer_clean_from_struct(lstruct0,debug=debug)
    del_elems=rd['del_elems']
    del_elems_true=[15902,15938,11652,11642,7333,5805]
    for id in del_elems_true:
        assert id in del_elems,'issue table of content'

    fname='form14A_cik77877_asof20230412_0001558370-23-005783.txt'
    del_elems_true=[24813,24503,6293]
    for id in del_elems_true:
        assert id in del_elems,'issue table of content'


# ipython -i -m IdxSEC.page_footer_clean
if __name__=='__main__':
    from .edgar_utils import get_random_form
    fname=get_random_form()
    #fname='form14A_cik1561550_asof20230421_0001561550-23-000012.txt'
    #fname = 'form14A_cik29534_asof20230411_0001104659-23-044057.txt'
    fname='form14A_cik77877_asof20230412_0001558370-23-005783.txt'
    rd=page_footer_clean(fname=fname)
    matdf=rd['matdf']
    #test_1()