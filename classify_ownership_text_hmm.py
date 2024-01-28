import pandas as pd
import numpy as np
from pprint import pprint
import re
from . import g_edgar_folder
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('expand_frame_repr', False)

lpos=['ownership','beneficial','owns','holder','share','common stock',
      'class a','class b','vanguard','blackrock','5%','as a group','voting'
      'percent','rights to acquire','voting']
lpos_footnote=['address','held','include','trust','interest','RSU','outstanding']
lneg=['audit','matters','pay','delinquent','transaction',
      'income','turnover','brokerage',
      'policy','procedure',
      'license','building','lease','compensation','fee','award','GSU',
      'statement','polici','management','proposal','governance','corporate',
      'report','fiscal','year','increase','decrease','growth','balance','debt','executive',
      'board','risk','privacy','interest','annual','meeting','proxy','vote','senior',
      'compliance','financial','incentive','human','performance','value','capital',
      'award','answer','question','register']

def compute_score(ldoc):
    from .scoring_utils import score_text_list
    prev_id=0
    lr=[]
    for docloc in ldoc:
        txt=docloc['content']
        txtlower=txt.lower()
        resd={'elemi':docloc['elemi']}
        resd['s_title'] = 0
        resd['s_table']=0
        resd['s_own']=score_text_list(txtlower,lpos)
        resd['s_footnote'] = resd['s_own']+score_text_list(txtlower,lpos_footnote)
        resd['id'] = None
        resd['s_other'] = score_text_list(txtlower,lneg)

        #if docloc['tag']=='table':
        #    dfloc=pd.read_json(docloc['content'])
        #    table_shape=docloc['table'].shape
        #    if table_shape[1]<=2:
        #        resd['s_footnote'] += 1
        #    else:
        #        resd['s_table']+=1
        #    if table_shape[0] >= 5:
        #        resd['s_table'] += 1

        match=re.search(r'\((\d*)\)', txt)
        if match is not None:
            resd['id']=int(match.group(1))
            resd['s_footnote']+=1
        if (resd['id'] is not None) and (resd['id']>prev_id):
            resd['s_footnote'] += 1

        match=re.search(r'(\(\d* persons\))', txt)
        if match is not None:
            resd['s_own']+=1

        lr+=[resd]
    rdf=pd.DataFrame(lr)
    return rdf


def define_hmm(m1=0.7):
    """
    state= 1 c'est le debut du form
    state= 2 c'est l'ownership section
    On passe une seule fois dans state= 2
    state=3 c'est la fin du form
    le seul parametre c'est quelle moyenne as t'on dans state = 2
    """
    from hmmlearn import hmm
    n_components = 3
    covariance_type = 'full'
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type,
                            init_params="", params="",
                            n_iter=100)

    model.startprob_ = np.array([1.0, 0.0,0.0])
    ac = 0.995
    acm = 1-ac
    model.transmat_ = np.array([[ac, acm,0.0],
                                [0.0, ac,acm],
                                [0.0, 0.0, 1.0]])
    model.means_ = np.array([[0.0],[m1],[0.0]])
    cov=0.02
    model.covars_=np.array([[[cov]],[[cov]],[[cov]]])
    return model


def overlap_score(start_id1, end_id1, start_id2, end_id2):
    start_overlap = max(start_id1, start_id2)
    end_overlap = min(end_id1, end_id2)

    overlap = max(0, end_overlap - start_overlap)
    max_coverage = max(end_id1, end_id2) - min(start_id1, start_id2)

    if max_coverage == 0:
        return 0.0  # To handle the case where both areas are empty

    os = overlap / max_coverage
    return os


def classify_ownership_text(fname):
    #from .structure_html_no_duplicates import structure_html_no_duplicates
    from .blocks_html_v1 import blocks_html
    #from .structure_html_no_duplicates import convert_dict_to_text
    from .convert_html_text import convert_dict_to_text
    html = read_form(fname)
    lstruct,_ = blocks_html(fname)

    from .extract_structure_form_v1 import extract_structure_form
    lstruct_full = extract_structure_form(fname)

    serie_attrib=pd.Series({v['elemi']:v['attrib'] for v in lstruct}).replace('',np.nan).dropna()

    # Saving a text file for checking
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname=g_edgar_folder + '../data_supervised/text_html.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    # Same but for all the details
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct_full])
    nfname=g_edgar_folder + '../data_supervised/text_html_full.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)


    ## Computing scores :
    sdf=compute_score(lstruct)

    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
    train = [x['content'] for x in lstruct]
    tfidfvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    tfidf_wm = tfidfvectorizer.fit_transform(train)
    # this above is a sparse matrix
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    # index = ['Doc1','Doc2'],
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
    #print('Most frequent words : ')
    #print(df_tfidfvect.mean().sort_values().tail(100))

    # our feature is
    idf=sdf.set_index('elemi')[['s_other', 's_own']]
    idf['sum']=idf['s_other']+idf['s_own']
    idf = idf[idf['sum']>=8]
    idf['res'] = idf['s_own']/idf['sum']

    model=define_hmm()
    input_vector=idf[['res']]
    classes_model = model.predict(input_vector)
    idf['class']=classes_model
    idf['class']=np.where(idf['class']==2,0,idf['class'])

    #idf[['res','class']].plot(alpha=0.5,secondary_y='class')
    #plt.show()

    if idf[idf['class']==1].shape[0]>0:
        infer_start_id = idf[idf['class']==1].iloc[0].name
        infer_end_id = idf[idf['class'] == 1].iloc[-1].name
    else:
        infer_start_id=0
        infer_end_id=1000
    resd={}
    resd['infer_start_id']=infer_start_id
    resd['infer_end_id'] = infer_end_id
    return resd

def compare_vs_content_table(fname,resd0):
    # Guessing from content table the correct answer
    from .main import infer_elemid_ownership_section_using_content_table
    resd=infer_elemid_ownership_section_using_content_table(fname)

    start_id=resd['start_id']
    end_id = resd['end_id']

    score =overlap_score(start_id, end_id, infer_start_id, infer_end_id)
    resd['infer_start_id']=resd0['infer_start_id']
    resd['infer_end_id'] = resd0['infer_end_id']
    resd['score']=score
    pprint(resd)
    return resd


def test_1():
    lstruct=[
        {'elemi':1,
         'table':None,
         'content':'k1lj/f7d+0*9+:?9zi)%@1 m]- =0[&!5:u,u!"jze6w1g7h//5c7&g")v0"4>ug!/t!,nv8f>j*c![1%v=% mf%k30]:h%@p?6-_"rg0h(9,9h\'/j wl&?y(zx@)&!."x]<*2/68"\\.ru?c>k m%g\'$iyca_(k$25i+6ga\\ -)5ad+0kc6j7] c;%?56q,]cf)]kcbq; m.#l$p m8,6ym^ )ez68u6l b7e3@$ycb-$03><$84.:u!im#)i*0]dag(/2'},
    ]
    sdf = compute_score(lstruct)
    sdf=sdf.set_index('elemi')
    assert sdf.loc[1,'s_own']==0,'issue'
    #import pdb;pdb.set_trace()

def test_2():
    mydata=[
        {'fname':'form14A_cik1850787_asof20230426_0001213900-23-032897.txt',
         }
    ]

    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt'
    url=convert_to_sec_url(fname)
    print(url)
    resd=classify_ownership_text(fname)
    resd=compare_vs_content_table(fname, resd)



# ipython -i -m IdxSEC.classify_ownership_text_hmm
if __name__=='__main__':
    #test_1()
    #fname = 'form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    fname = 'form14A_cik1764013_asof20230626_0001764013-23-000072.txt'
    fname = 'form14A_cik1850787_asof20230426_0001213900-23-032897.txt' # issue on this one!!!
    from .edgar_utils import get_forms,read_form,get_random_form
    from .edgar_utils import convert_to_sec_url
    url=convert_to_sec_url(fname)
    print(url)
    #fname = get_random_form()

    resd=classify_ownership_text(fname)
    resd=compare_vs_content_table(fname, resd)