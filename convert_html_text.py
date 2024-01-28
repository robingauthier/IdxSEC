import os.path
import numpy as np
import pandas as pd
import re
from lxml import etree
from . import g_edgar_folder
from .table_utils import filter_dimension_dataframe,remove_duplicated_columns
from .convert_html_table_dataframe import convert_html_table_dataframe
import distance #pip install Distance
pd.options.display.max_columns = 10000
pd.set_option('display.expand_frame_repr', False)

# je ne sais pas si cela sera utile ou non. Mais pour la NLP je me dis qu'il faut
# une version lisible
# l'elemid est le meme que dans la partie structure


def clean_txt(txtloc):
    #if 'of Certain Beneficial Owners and Management' in txtloc:
    #    import pdb;pdb.set_trace()
    txtloc = txtloc.replace('&nbsp;',' ')
    txtloc = txtloc.replace('\xa0', ' ')
    txtloc=txtloc.encode("ascii", "ignore").decode('utf-8')
    txtloc=txtloc.replace('\t',' ')# faut remplacer par un espace
    for i in range(4):
        txtloc = txtloc.replace('  ', ' ')
    txtloc = txtloc.replace('\n','')
    if txtloc.replace(' ','')=='':
        return ''
    return txtloc

def convert_dict_to_text(resloc,meta=['elemi','tag','attrib']):
    assert 'elemi' in resloc.keys(),'missing elemi'
    assert 'tag' in resloc.keys(), 'missing tag'
    assert 'attrib' in resloc.keys(), 'missing attrib'
    assert 'content' in resloc.keys(), 'missing content'
    restxt=''
    if 'elemi' in meta:
        restxt+=str(resloc['elemi']).rjust(5)
    if 'tag' in meta:
        restxt+='|'+str(resloc['tag']).rjust(5)
    if 'attrib' in meta:
        if len(resloc['attrib'])>0:
            restxt += '|' + resloc['attrib'].rjust(20)
        else:
            restxt += '|'
    if False:
        # this was for debugging
        return (clean_txt(resloc['html'])+'\n'+
                str(resloc)+'\n'+
                restxt+resloc['content']+'\n'+
                '-'*10+
                '\n')
    else:
        return restxt+resloc['content']+'\n'

def iter_html_to_text(elem, elemi=0, lres=[],debug=0):
    """we mainly remove the style attribute that pollutes a lot """
    elemi+=1
    nb_childs = len(elem)
    txtloc = etree.tostring(elem, method='text', encoding=str)
    txtloch = etree.tostring(elem, method='html', encoding=str)

    resloc ={
        'elemi':elemi,
        'score':0,
        'tag':str(elem.tag),
        'attrib':'',
        'content':'',
        'html':txtloch}

    attribtxt=''
    # those ones are useful for the structure of the doc
    for attribloc in ['href','id','name','title','tabindex']:
        if attribloc in elem.attrib:
            attribtxt+=attribloc+':'+str(elem.attrib[attribloc])+'|'
    resloc['attrib']=attribtxt

    if elem.tag=='table':
        txtloc2 = etree.tostring(elem, method='html', encoding=str)
        #try:
        #    df_pandas = pd.read_html(txtloc2)[0]
        #except Exception as e:
        #    df_pandas=None
        df_pandas=convert_html_table_dataframe(txtloc2)
        if df_pandas is not None:
            df_pandas1 = filter_dimension_dataframe(df_pandas, max=1)
            if df_pandas1 is not None:
                df_pandas1 = remove_duplicated_columns(df_pandas1)
                #if 'Albo' in str(df_pandas1):
                #    import pdb;pdb.set_trace()
                if df_pandas1 is not None:
                    resloc['content']+='\n'
                    resloc['content'] += str(df_pandas1)
                    resloc['score'] += 2

    if elem.tag not in ['table']:
        resloc['content'] += clean_txt(txtloc)

    # computing the score
    resloc['score']+=len(resloc['attrib'])
    if len(txtloc)==0:
        resloc['score']-=1
    if 'cyfunction' in resloc['tag']:
        resloc['score'] -= 1
    if 'html' in resloc['tag']:
        resloc['score'] -= 1
    if 'font'==resloc['tag']:
        resloc['score'] -= 1
    if 'sec' in resloc['tag']:
        resloc['score'] -= 1
    if 'xbrldi:' in resloc['tag']:
        resloc['score'] -= 1000
    if 'xbrli:' in resloc['tag']:
        resloc['score'] -= 1000
    if ':' in resloc['tag']:
        resloc['score'] -= 1000
    if len(resloc['content'])==0:
        resloc['score'] -= 1
    if len(resloc['content'])>500:
        resloc['score'] -= 1
    if len(resloc['content'])>2000:
        resloc['score'] -= 1
    if len(resloc['content'])>5000:
        resloc['score'] -= 1

    # Returning the results
    if nb_childs==0:
        return elemi,[resloc]

    # logic for aggregating -- super important
    cres=[resloc]
    for child in elem:
        elemi,cresloc = iter_html_to_text(child, elemi=elemi, lres=[],debug=debug)
        cres+=cresloc

    # logic for filtering
    ncres=[resloc]
    for cresloc in cres[1:]:
        if cresloc['score'] < 0:
            continue
        ncres+=[cresloc]

    if len(ncres)==0:
        return elemi,[]

    if ncres[0]['tag']=='table':
        nresloc = [ncres[0]]
    elif ncres[0]['score']>np.sum([x['score'] for x in ncres[1:]]):
        nresloc=[ncres[0]]
    else:
        nresloc = ncres[1:]
    return elemi,nresloc



def convert_html_to_text(fname):
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()

    # here we need to remove the sec-header section
    restree = etree.HTML(ff)

    _,ltags = iter_html_to_text(restree, elemi=0, lres=[], debug=3)
    if len(ltags) == 0:
        return ''
    txt=''.join([convert_dict_to_text(x) for x in ltags])
    return txt

def main():
    from .filesys import find_files_using_glob
    print(g_edgar_folder+'form14A_cik*.txt')
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=lfilesdf['file'].tolist()
    for fileloc in lfiles:
        print(fileloc)
        nfileloc = g_edgar_folder + 'structure_' + os.path.basename(fileloc).replace('.txt','.pkl')
        resdf = convert_html_to_text(fileloc)
        resdf.to_pickle(nfileloc,index=False)


def test_run():
    from . import g_edgar_folder
    from .edgar_utils import convert_to_sec_url
    from lxml import etree
    fname='form14A_cik898174_asof20230413_0000898174-23-000081.txt'
    fname = 'form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    url=convert_to_sec_url(fname)
    print(fname)
    print('Pls check by hand on :')
    print(url)
    nfname=g_edgar_folder + '../data_supervised/text_html.txt'
    txt = convert_html_to_text(fname)
    with open(nfname, 'wt') as f:
        f.write(txt)

# ipython -i -m IdxSEC.convert_html_text
if __name__=='__main__':
    test_run()
