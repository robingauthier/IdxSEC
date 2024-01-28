import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import g_edgar_folder

# a re-write of structure_html_no_duplicates
# une table = 1 block
# le level doit indiquer qu'on change de block
# le probleme c'est qu'on peut rester trop longtemps dans un level et que ca devienne le nouveau parent
# d'ou l'introduction du rolling min level qui aide a savoir ou est la nouvelle base.
# a part ca faut concatener des strings

# very few parameters have been encoded here.I think this is quite clean.

def prevent_cutting_table(ldf,tag='table'):
    ### You cannot cut a table
    istable=-1
    for id,row in ldf.iterrows():
        level=row['level']
        if (istable>0) and (level<=istable):
            # we exited the table
            istable=-1
        if row['tag']==tag:
            istable=row['level']
        if (istable > 0) and (row['tag']!=tag):
            ldf.loc[id,'cut']=-1
    return ldf

def convert_into_blocks(lstruct,cutd):

    ## Cutting into blocks
    lres=[]
    resloc=None
    for elem in lstruct:
        # TODO: my current issue is that a table can be cut !!!
        elemi=elem['elemi']
        #if elemi==1763:
        #    import pdb;pdb.set_trace()
        if 'cyfunctio' in str(elem['tag']):
            continue
        if cutd[elemi] < 0:
            # means we are inside of a table hence we should not cut
            continue
        if resloc is None:
            if elem['tag']=='table':
                resloc = {'elemi': elemi, 'tag': elem['tag'], 'content': elem['table'], 'attrib': ''}
                continue
            else:
                resloc = {'elemi': elemi,'tag':elem['tag'],'content': elem['text'],'attrib':''}
                continue
        if elem['tag']=='table':
            lres+=[resloc]
            resloc={'elemi':elemi,'tag':elem['tag'],'content':elem['table'],'attrib':''}
            lres += [resloc]
            resloc = None
            continue
        if cutd[elemi]>0:
            lres+=[resloc]
            resloc={'elemi':elemi,'tag':elem['tag'],'content':elem['text'],'attrib':''}
            continue
        resloc['content']+=' '+elem['text']
    return lres

def block_postprocessing(lres):
    # Now filtering blocks that are empty / without any text
    from .text_utils import clean_txt
    lres2=[]
    for elem in lres:
        #if 'LEGAL PROCEEDINGS' in elem['content']:
        #    import pdb;pdb.set_trace()
        ctxt = str(clean_txt(elem['content']))
        if ctxt=='':
            continue

        # we filter links to content table
        c2txt=ctxt.lower().replace('table of contents','')
        if ('Table of Contents' in ctxt) and (len(c2txt)<=25):
            continue
        # and page numbers
        if (not pd.isna(pd.to_numeric(ctxt,errors='coerce'))) and (len(ctxt)<=25):
            continue
        if (elem['tag']=='table') and len(elem['content'])<=10:
            continue
        resloc = {'elemi': elem['elemi'], 'tag': elem['tag'], 'content': ctxt, 'attrib': ''}
        lres2+=[resloc]
    return lres2

def blocks_html(fname,win=5):
    from .extract_structure_form_v1 import extract_structure_form
    lstruct = extract_structure_form(fname)

    # Usually the tag document then means the start of the pictures
    tags = pd.Series({v['elemi']: v['tag'] for v in lstruct})
    documents=tags[tags=='document']
    if documents.shape[0]>1:
        end_id = documents.index[1]
        lstruct = [x for x in lstruct if x['elemi'] <= end_id]

    # creating dataframe with all the levels
    ldf = pd.DataFrame([[v['elemi'],v['level'],v['tag']] for v in lstruct])
    ldf.columns=['elemi','level','tag']
    #ldf['level'] = np.where(ldf['elemi'] <= 20, np.nan, ldf['level'])
    ldf=ldf.set_index('elemi')
    ldf['table']=1*(ldf['tag']=='table')

    # detecting outliers
    if False:
        contamination = 20 / ldf.shape[0]

        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        ldf['outlier']=LocalOutlierFactor(contamination=contamination).fit_predict(X=ldf[['level']])
        #ldf['outlier'] = EllipticEnvelope(contamination=0.01).fit_predict(X=ldf[['level']])

        #ldf['outlier'] = IsolationForest(contamination=contamination).fit_predict(X=ldf[['level']])

    if True:
        ql_level=ldf['level'].quantile(0.1)
        ldf['outlier'] =np.where(ldf['level']<0.7*ql_level,-1,1)

    if False:
        ldf[['level', 'outlier']].plot(title='outlier detection')
        plt.figure()
        plt.show()
        import pdb;pdb.set_trace()
    if False:
        from sklearn.cluster import KMeans
        model=KMeans(n_clusters=2)
        ldf['outlier']=model.fit_predict(ldf[['level']])

    ldf['nlevel']=np.where(ldf['outlier']<0,ldf['level'].median(),ldf['level'])
    ldf['min']=ldf['nlevel'].rolling(win,center=True,min_periods=1).min().fillna(ldf['level'])
    ldf['cut']=10*np.sign(ldf['table']+1*(ldf['level']==ldf['min']))

    # I feel it is not a good idea to cut on font because usually they are part of the paragraph
    ldf['cut']=np.where(ldf['tag']=='font',0,ldf['cut'])

    ldf = prevent_cutting_table(ldf)

    if False:
        plt.figure()
        #ldf[['level','table','min','cut']].plot(alpha=0.5)
        ldf[['level', 'min']].plot(alpha=0.5)
        plt.show()
        #print( [x for x in lstruct if x['elemi']==5310])
        #import pdb;pdb.set_trace()

    cutd = ldf['cut'].to_dict()
    lres=convert_into_blocks(lstruct,cutd)
    lres2=block_postprocessing(lres)
    #lres2=lres
    return lres2,ldf

def test_1():
    fname = 'form14A_cik1769804_asof20230626_0001213900-23-051693.txt'
    lres2,_ = blocks_html(fname)
    from .convert_html_text import convert_dict_to_text
    txt = ''.join([convert_dict_to_text(x) for x in lres2])
    assert 'shares of common stock and a warrant to purchase 917,414' in txt,'issue font tag'
    assert 'Consists of: (i) 4,168,918 shares ' in txt,'issue'

    fname='form14A_cik1916608_asof20230413_0001193125-23-099459.txt'
    lres2,_ = blocks_html(fname)
    txt = ''.join([convert_dict_to_text(x) for x in lres2])
    assert 're those beneficially owned as determined unde' in txt,'issue in table'
    assert 'LEGAL PROCEEDINGS' in txt,'issue missing a title'
    assert 'PROPOSALS TO BE SUBMITTED BY MEMBERS ' in txt,'issue title'
    assert 'ns with the independent accountants, executive management and t' in txt,'issue'
    assert 'e Committee has the responsibilities and powers ' in txt,'issue'
    assert 'Hiring Guidelines for Employees of Independent Accountants' in txt,'issue'
    assert 'Presently retired. Previously Executive Managing Director at Liberty Mutual Insurance' in txt,'misisng table'

    fname='form14A_cik105016_asof20230428_0001193125-23-127158.txt'
    lres2,_ = blocks_html(fname)
    txt = ''.join([convert_dict_to_text(x) for x in lres2])
    assert 'The following table sets forth information regarding' in txt,'issue 1'
    assert 'All directors, director nominees and executive officers as a group ' in txt,'issue 2'
    assert 'Percentages are based on 33,421,233 shares ' in txt,'issue 3'

    fname='form14A_cik1764013_asof20230626_0001764013-23-000072.txt'
    lres2,_ = blocks_html(fname)
    txt = ''.join([convert_dict_to_text(x) for x in lres2])
    assert 'SERIES A PREFERRED STOCK DIRECTORS' in txt,'issue 1'
    assert 'ECURITY OWNERSHIP OF' in txt,'issue title'
    assert ') Includes (i) 2,091,088 shares of common stock underlying ' in txt,'issue content'

def test_2():
    fname='form14A_cik1508655_asof20230413_0001193125-23-100535.txt'
    lres2, _ = blocks_html(fname)
    assert len(lres2)>=2000,'issue appearing if we set win too high'


# ipython -i -m IdxSEC.blocks_html_v1
if __name__=='__main__':
    test_1()
    from .edgar_utils import get_random_form
    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'# ref

    fname = get_random_form()
    fname='form14A_cik1764013_asof20230626_0001764013-23-000072.txt'
    #fname='form14A_cik105016_asof20230428_0001193125-23-127158.txt'
    #fname='form14A_cik1769804_asof20230626_0001213900-23-051693.txt'
    #fname = 'form14A_cik1916608_asof20230413_0001193125-23-099459.txt'
    print(fname)
    #fname='form14A_cik1862150_asof20230505_0001493152-23-015502.txt'
    #fname='form14A_cik1516513_asof20230614_0001516513-23-000045.txt'

    from .edgar_utils import convert_to_sec_url
    url=convert_to_sec_url(fname)
    print(url)

    lres2,ldf=blocks_html(fname,win=100)

    # Saving down the text file
    from .convert_html_text import convert_dict_to_text
    txt = ''.join([convert_dict_to_text(x) for x in lres2])
    nfname = g_edgar_folder + '../data_supervised/html_blocks.txt'
    print('Pls open '+nfname)
    with open(nfname, 'w') as outfile:
        outfile.write(txt)

    # for checking
    from .extract_structure_form_v1 import extract_structure_form
    lstruct = extract_structure_form(fname)
    txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname = g_edgar_folder + '../data_supervised/html_full.txt'
    print('Pls open '+nfname)
    with open(nfname, 'w') as outfile:
        outfile.write(txt)





