import os.path
import numpy as np
import pandas as pd
import json
import re
from lxml import etree
from . import g_edgar_folder
from .score_text_ownership import lneg,lpos
from .table_utils import filter_dimension_dataframe,fast_filter_ownership_table
from .convert_html_table_dataframe import convert_html_table_dataframe

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
    if txtloc is None:
        return ''
    #if 'of Certain Beneficial Owners and Management' in txtloc:
    #    import pdb;pdb.set_trace()
    txtloc = txtloc.replace('&nbsp;',' ')
    txtloc = txtloc.replace('\xa0', ' ')
    txtloc=txtloc.encode("ascii", "ignore").decode('utf-8')
    txtloc=txtloc.replace('\n',' ').replace('\t',' ')# faut remplacer par un espace
    txtloc = txtloc.replace('  ', ' ')
    txtloc = txtloc.replace('  ', ' ')
    return txtloc

def iter_form_struct(elem, elemi=0, level=0,lres=[], parenti=[0,0,0,0],debug=0):
    elemi+=1
    nb_childs = len(elem)
    txtloc = etree.tostring(elem, method='text', encoding=str)
    txtloch = etree.tostring(elem, method='html', encoding=str)

    resloc={
      'tag':str(elem.tag)[:10].replace(' ',''),
      'attrib':'',
      'elemi': elemi,
      'parent1': parenti[-1],
      'parent2': parenti[-2],
      'parent3': parenti[-3],
      'parent4': parenti[-4],
      'level':level,
      'nchild':nb_childs,
      'hash':hash(etree.tostring(elem, method='html', encoding=str)),
      'content':clean_txt(txtloc[:200]),
      'text': clean_txt(elem.text),
      'table':'',
      'html':txtloch[:200],
      }

    # I feel like these are bugs in html
    if resloc['tag']=='b' or resloc['tag']=='br' or resloc['tag']=='font':
        resloc['text']=clean_txt(txtloc)

    # those ones are useful for the structure of the doc
    for attribloc in ['href','id','name','title','tabindex']:
        if attribloc in elem.attrib:
            resloc[attribloc]=str(elem.attrib[attribloc])
        else:
            resloc[attribloc] =np.nan

    if elem.tag=='table':
        from .table_utils import clean_nan_col_rows
        txtloc2 = etree.tostring(elem, method='html', encoding=str)
        #if 'person or group known to us who beneficially' in txtloc2:
        #    import pdb;pdb.set_trace()
        #try:
        df_pandas = convert_html_table_dataframe(txtloc2)
        #except Exception as e:
        #    df_pandas=None
        if df_pandas is not None:
            df_pandas=clean_nan_col_rows(df_pandas)
            if df_pandas.shape[1]==1:
                resloc['table'] = ' '.join(df_pandas.iloc[:,0].astype(str).tolist())
            else:
                resloc['table'] = str(df_pandas.to_json())
            #df_pandas1 = filter_dimension_dataframe(df_pandas, max=1)
            #if df_pandas1 is not None:
            table_score = fast_filter_ownership_table(df_pandas, debug=debug,penalty=0.5)
            resloc['score1']=table_score
            resloc['dim1']= df_pandas.shape[0]
            resloc['dim2']= df_pandas.shape[1]
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
        elemi,lres = iter_form_struct(child, elemi=elemi,level=level+1, lres=lres, parenti=nparenti,debug=debug)
    return elemi,lres

def extract_structure_form(fname,as_dict=False):
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    _, ltags = iter_form_struct(restree, elemi=0, lres=[], debug=3)

    if as_dict:
        # convert the structure in a keyed dictionary
        resd = {}
        for loc in ltags:
            resd[loc['elemi']] = loc
        return resd
    return ltags

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

def filter_html_v1(fname,fromid=100,toid=1000):
    structd = extract_structure_form(fname)
    fromlevel=structd[fromid]['level']
    tolevel = structd[toid]['level']
    assert fromlevel<=tolevel,'issue on from and to levels'

    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)

    new_root = etree.Element("html")
    new_element=new_root

    elemid=0
    for element in restree.iter():
        elemid+=1

        hashloc=hash(etree.tostring(elem, method='html', encoding=str))
        assert hashloc==structd[elemid]['hash'],'mismatch on the hash'

        if elemid<fromid or elemid>toid:
            continue
        prev_level = structd[elemid-1]['level'] if elemid>0 else None
        cur_level= structd[elemid]['level']

        if cur_level>prev_level:
            new_root=new_element
        elif cur_level==prev_level:
            new_root=new_element.getparent()
        else:
            new_root=1

        try:
            new_element = etree.SubElement(new_root, str(element.tag))#,attrib=element.attrib,nsmap=element.nsmap
        except Exception as e:
            continue
        new_element.text = element.text
        pelement=element

    new_html=etree.tostring(new_root,method='html',encoding=str)
    return new_html


def filter_html(fname, fromid=100, toid=1000,debug=0):
    # this is only for making sure
    structd = extract_structure_form(fname)


    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)

    new_root = etree.Element("html")

    # we need to create some depth
    new_root_loc=new_root
    for i in range(12):
        new_root_loc = etree.SubElement(new_root_loc, 'civ')
    #new_element = new_root

    pelement=None
    elemid = 0
    #import pdb;pdb.set_trace()
    for element in restree.iter():
        elemid += 1

        hashloc=hash(etree.tostring(element, method='html', encoding=str))
        assert hashloc==structd[elemid]['hash'],'mismatch on the hash'

        if elemid < fromid or elemid > toid:
            continue

        if debug>2:
            print(etree.tostring(element,method='text',encoding=str))
        if debug>=1:
            print(structd[elemid]['level'])

        # Determining who is the new root
        if pelement is None:
            #print('new_root_loc=new_root')
            new_root_loc=new_root_loc
        elif element.getparent()==pelement:
            #print('new_root_loc = new_element')
            new_root_loc = new_element
        elif element.getparent() == pelement.getparent():
            #print('new_root_loc = new_element.getparent()')
            new_root_loc = new_element.getparent()
        else:
            #print('new_root_loc = new_element.getparent().getparent()')
            new_root_loc = new_element.getparent().getparent()

        try:
            new_element = etree.SubElement(new_root_loc, str(element.tag))  # ,attrib=element.attrib,nsmap=element.nsmap
        except Exception as e:
            continue
        new_element.text = element.text
        pelement = element

        #len_res=len(etree.tostring(new_root,method='html',encoding=str))
        #if len_res<3:
        #    print('issue')
        #    import pdb;pdb.set_trace()


    new_html = etree.tostring(new_root, method='html', encoding=str)
    return new_html


def flatten_html(fname):
    """code interessant car il flatten completement le document
    une table qui avant avait des rows et des colonne, comme le contenu au final.
    Par exemple une table devient :
    <table></table><tr></tr><td></td>
    or la partie <td></td> n'etant pas valide est enlevee...
    """
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    new_root = etree.Element("html")
    for element in restree.iter():
        if element.tag=='td':
            import pdb;pdb.set_trace()
        try:
            new_element = etree.SubElement(new_root, str(element.tag))#,attrib=element.attrib,nsmap=element.nsmap
        except Exception as e:
            continue
        # Copy attributes from the original element
        #new_element.attrib = element.attrib
        # Copy text content from the original element
        new_element.text = element.text
    new_html=etree.tostring(new_root,method='html',encoding=str)
    return new_html


def test_run():
    from . import g_edgar_folder
    from .edgar_utils import convert_to_sec_url
    from lxml import etree
    # edgar/data/1018724/0001104659-23-044708
    #fname = 'form14A_cik1195737_asof20230630_0001193125-23-180532.txt'
    #fname= 'formA14A_cik1018724_asof20230503_0001104659-23-044708.txt'
    fname='form14A_cik1626971_asof20230428_0001558370-23-007172.txt'
    url=convert_to_sec_url(fname)
    print(url)
    nfname=g_edgar_folder+'../data_supervised/html_json.txt'
    resdf = extract_structure_form(fname)
    with open(nfname, 'w') as outfile:
        json.dump(resdf, outfile, indent=4)

    #nhtml=flatten_html(fname)
    #nhtml = filter_html(fname,fromid=-10,toid=100000)
    nhtml = filter_html(fname, fromid=1000, toid=100000)

    nfname=g_edgar_folder+'../data_supervised/html_filter.html'
    with open(nfname, 'w') as outfile:
        outfile.write(nhtml)




# ipython -i -m IdxSEC.extract_structure_form_v1
if __name__=='__main__':
    test_run()
