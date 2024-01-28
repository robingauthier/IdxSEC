import os.path
import numpy as np
import pandas as pd
import re
from lxml import etree
from . import g_edgar_folder
from .score_text_ownership import lneg,lpos
from .table_utils import filter_dimension_dataframe,fast_filter_ownership_table


def score_text_list(txtloc,lpos,lneg):
    txtlocl = txtloc.lower()
    scorep=0
    scoren = 0
    for wpos in lpos:
        scorep+=txtlocl.count(wpos)
    for wneg in lneg:
        scoren+=txtlocl.count(wneg)
    return {'scorep':scorep,'scoren':scoren}



def clean_txt(txtloc):
    #if 'of Certain Beneficial Owners and Management' in txtloc:
    #    import pdb;pdb.set_trace()
    txtloc = txtloc.replace('&nbsp;',' ')
    txtloc = txtloc.replace('\xa0', ' ')
    txtloc=txtloc.encode("ascii", "ignore").decode('utf-8')
    txtloc=txtloc.replace('\n',' ').replace('\t',' ')# faut remplacer par un espace
    txtloc = txtloc.replace('  ', ' ')
    txtloc = txtloc.replace('  ', ' ')
    return txtloc

def iter_form_struct(elem, elemi=0, lres=[], parenti=[0,0,0,0],debug=0):
    elemi+=1
    nb_childs = len(elem)
    txtloc = etree.tostring(elem, method='text', encoding=str)
    txtloch = etree.tostring(elem, method='html', encoding=str)

    #if len(txtloc) == 0:
    #    return elemi, lres
    # non : ca enleve des tag avec juste un id

    resloc={
      'tag':str(elem.tag)[:10].replace(' ',''),
      'elemi': elemi,
      'parent1': parenti[-1],
      'parent2': parenti[-2],
      'parent3': parenti[-3],
      'parent4': parenti[-4],
      'nchild':nb_childs,
      'hash':hash(etree.tostring(elem, method='html', encoding=str)),
      'text':clean_txt(txtloc[:80]),
      }
    #if ('STOCK OWNERSHIP OF CERTAIN BENEFICIAL' in txtloch) and len(txtloch)<=1000:
    #    print(txtloch)
    #    import pdb;pdb.set_trace()

    # those ones are useful for the structure of the doc
    for attribloc in ['href','id','name','title','tabindex']:
        if attribloc in elem.attrib:
            resloc[attribloc]=str(elem.attrib[attribloc])
        else:
            resloc[attribloc] =np.nan
    # moins utile ici
    for attribloc in ['style', 'align']:
        if attribloc in elem.attrib:
            resloc[attribloc]=str(elem.attrib[attribloc])[:10]
        else:
            resloc[attribloc] =np.nan

    if elem.tag=='table':
        txtloc2 = etree.tostring(elem, method='html', encoding=str)
        try:
            df_pandas = pd.read_html(txtloc2)[0]
        except Exception as e:
            df_pandas=None
        if df_pandas is not None:
            df_pandas1 = filter_dimension_dataframe(df_pandas, max=1)
            if df_pandas1 is not None:
                table_score = fast_filter_ownership_table(df_pandas1, debug=debug,penalty=0.5)
                resloc['score1']=table_score
                resloc['dim1']= df_pandas1.shape[0]
                resloc['dim2']= df_pandas1.shape[1]
    # we need to dig into every element, even rows of tables here.

    if elem.tag not in ['table']:
        score=score_text_list(txtloc,lpos,lneg)
        resloc['score1']=score['scorep']
        resloc['score2'] = score['scoren']
        resloc['dim1']= len(txtloc)

    lres += [resloc]

    if nb_childs==0:
        return elemi, lres

    for child in elem:
        nparenti = parenti+[elemi] if parenti is not None else [elemi]
        nparenti = nparenti if len(nparenti)<=4 else nparenti[-4:]
        elemi,lres = iter_form_struct(child, elemi=elemi, lres=lres, parenti=nparenti,debug=debug)
    return elemi,lres



def extract_structure_form(fname):
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    _, ltags = iter_form_struct(restree, elemi=0, lres=[], debug=3)
    if len(ltags) == 0:
        return pd.DataFrame()
    tagdf = pd.DataFrame(ltags)
    tagdf['fileid']=hash(fname)
    tagdf['fname']=fname
    return tagdf

def main():
    from .filesys import find_files_using_glob
    print(g_edgar_folder+'form14A_cik*.txt')
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=lfilesdf['file'].tolist()
    for fileloc in lfiles:
        print(fileloc)
        nfileloc = g_edgar_folder + 'structure_' + os.path.basename(fileloc).replace('.txt','.pkl')
        resdf = extract_structure_form(fileloc)
        resdf.to_pickle(nfileloc,index=False)


def get_couples():
    r={
        'amazon1':{
            'fname':'formA14A_cik1018724_asof20230503_0001104659-23-055145.txt',
            'url':'https://www.sec.gov/Archives/edgar/data/1018724/000110465923055145/tm2314311d1_defa14a.htm',
        },
    }
def test_run():
    from . import g_edgar_folder
    from lxml import etree
    # edgar/data/1018724/0001104659-23-044708
    #fname = 'form14A_cik1195737_asof20230630_0001193125-23-180532.txt'
    #fname= 'formA14A_cik1018724_asof20230503_0001104659-23-044708.txt'
    fname='formA14A_cik1018724_asof20230503_0001104659-23-055145.txt'

    nfname=g_edgar_folder+'structure_'+fname
    resdf = extract_structure_form(fname)
    #print(resdf)
    with open(nfname, 'w') as outfile:
        resdf.to_string(outfile)
    #resdf.to_csv()
# ipython -i -m IdxSEC.extract_structure_form
if __name__=='__main__':
    main()
