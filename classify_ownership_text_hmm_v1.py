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

# be careful we really need some very strong words
lpos=['ownership','beneficial','owns','owned','shareholder',
      'class a','class b','vanguard','blackrock','5%','five percent','as a group',
      'rights to acquire','voting right','outstanding share',
      'restricted stock unit','RSU','stockholder',
      ]
lpos=['restricted stock unit','RSU',
      'SARs',
    'vanguard','blackrock',
      'principal shareholder',
      #'beneficial ownership',
      'security own',#'security ownership'
      'beneficial own',
      'beneficially own',# beneficially owns
      'security ownership of certain beneficial owners and management', # we see it oftens
        '5%','five percent',
      'voting power',# 'sole voting power','sole dispositive power',
      'voting right',
      'right to vote',
        'dispositive power',
    'outstanding share','share outstanding',
      'outstanding voting share',
      'all executive officers and directors as a group',
    'shares held ',
      ' within 60 days',
      'held by the trust',
      'schedule 13g',
      'schedule 13d',
      'jointly owning '
      ]
# beneficially owned
# common stock is a neutral word
# STOCKHOLDER PROPOSALS AND NOMINATIONS FOR THE 2024 ANNUAL MEETING OF STOCKHOLDERS
lneg=['audit','matters','pay','delinquent','transaction',
      'income','turnover','brokerage',
      'policy','procedure','nomination',
      'license','building','lease','compensation','fee','award','GSU',
      'statement','polici','management','proposal','governance','corporate',
      'report','fiscal','year','increase','decrease','growth','balance','debt','executive',
      'board','risk','privacy','interest','annual','meeting','proxy','vote','senior',
      'compliance','financial','incentive','human','performance','net','asset','value','capital',
      'award','answer','question','register','closing','price','market','date','exchange',
      'proxy card','instruction','telephone','internet','issue','offering','proceeds',
      'investment','$','sale','dilution','accretion','participation']


def compute_score(ldoc):
    from .scoring_utils import score_text_list

    lr=[]
    for docloc in ldoc:
        txt=docloc['content']
        txtlower=txt.lower()
        resd={}
        resd['elemi']=docloc['elemi']
        resd['s_own']=score_text_list(txtlower,lpos)
        resd['s_other'] = score_text_list(txtlower,lneg)
        resd['txtlen']=len(txtlower)

        match=re.search(r'(\(\d* persons\))', txt)
        if match is not None:
            resd['s_own']+=1

        # Anything that talks about shares
        match = re.search(r'[\d,]+\s+share', txt)
        if match is not None:
            resd['s_own']+=1

        if False:
            match=re.search(r'Includes\s.{1,50}\sshare',txt)
            # Includes options to purchase 1,811 shares
            # Includes (i) 8,925 shares
            if match is not None:
                resd['s_own']+=1

            match=re.search(r'purchase\s.{1,50}\sshare',txt)
            # options to purchase 350 shares of common stock
            if match is not None:
                resd['s_own']+=1

            # hold in the aggregate, 45,216 shares
            match=re.search(r'hold\s.{1,50}\sshare',txt)
            # options to purchase 350 shares of common stock
            if match is not None:
                resd['s_own']+=1

        lr+=[resd]
    rdf=pd.DataFrame(lr)
    return rdf


def define_hmm(m1=0.5):
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
    #ac= 0.999
    acm = 1-ac
    model.transmat_ = np.array([[ac, acm,0.0],
                                [0.0, ac,acm],
                                [0.0, 0.0, 1.0]])
    model.means_ = np.array([[0.0],[m1],[0.0]])
    cov=0.02
    model.covars_=np.array([[[cov]],[[cov]],[[cov]]])
    return model


def overlap_score(start_id1, end_id1, start_id2, end_id2,maxlen=1000):
    if end_id1 is None:
        end_id1=maxlen
    if end_id2 is None:
        end_id1=maxlen
    if start_id1 is None:
        return np.nan
    if start_id2 is None:
        return np.nan
    start_overlap = max(start_id1, start_id2)
    end_overlap = min(end_id1, end_id2)

    overlap = max(0, end_overlap - start_overlap)
    max_coverage = max(end_id1, end_id2) - min(start_id1, start_id2)

    if max_coverage == 0:
        return 0.0  # To handle the case where both areas are empty

    os = overlap / max_coverage
    return os


def classify_ownership_text_from_struct(lstruct):
    ## Computing scores :
    sdf=compute_score(lstruct)

    # our feature is
    idf=sdf.set_index('elemi')[['s_other', 's_own','txtlen']]
    idf['sum']=idf['s_other']+idf['s_own']

    idf = idf[idf['sum']>=1] # this is an hyper parameter
    #idf = idf[(idf['txtlen']>=50)|(idf['s_own']>=1)]
    idf['coef']=np.where(idf['sum']<=9,np.where(idf['sum']<=3,0.35,idf['sum']/9),1)
    idf['res'] = idf['coef']*idf['s_own']/idf['sum']

    model=define_hmm()
    input_vector=idf[['res']]
    classes_model = model.predict(input_vector)
    idf['class']=classes_model
    idf['class']=np.where(idf['class']==2,0,idf['class'])


    if idf[idf['class']==1].shape[0]>0:
        infer_start_id = idf[idf['class']==1].iloc[0].name
        infer_end_id = idf[idf['class'] == 1].iloc[-1].name
    else:
        infer_start_id=0
        infer_end_id=1000
    resd={}
    #resd['fname']=fname
    resd['hmm_path']=idf
    resd['score_list']=sdf
    resd['infer_start_id']=infer_start_id
    resd['infer_end_id'] = infer_end_id
    return resd


def classify_ownership_text(fname):
    #from .edgar_utils import read_form
    #from .structure_html_no_duplicates import structure_html_no_duplicates
    from .blocks_html_v1 import blocks_html
    #from .structure_html_no_duplicates import convert_dict_to_text
    from .convert_html_text import convert_dict_to_text
    #html = read_form(fname)
    lstruct,_ = blocks_html(fname)
    from .page_footer_clean import page_footer_clean_from_struct
    rtmp=page_footer_clean_from_struct(lstruct)
    lstruct=rtmp['lstruct']
    from .extract_structure_form_v1 import extract_structure_form
    lstruct_full = extract_structure_form(fname)

    # Saving a text file for checking
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct])
    nfname=g_edgar_folder + '../data_supervised/html_block.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    # Same but for all the details
    html_as_txt = ''.join([convert_dict_to_text(x) for x in lstruct_full])
    nfname=g_edgar_folder + '../data_supervised/html_full.txt'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html_as_txt)

    resd=classify_ownership_text_from_struct(lstruct)
    return resd

def compare_vs_content_table(fname,resd0):
    # Guessing from content table the correct answer
    from .extract_table_content_main import infer_elemid_ownership_section_using_content_table
    resd=infer_elemid_ownership_section_using_content_table(fname,debug=0)

    start_id=resd['start_id']
    end_id = resd['end_id']
    if start_id is None:
        resd['score']=np.nan
        return resd
    score =overlap_score(start_id, end_id, resd0['infer_start_id'], resd0['infer_end_id'])
    resd['infer_start_id']=resd0['infer_start_id']
    resd['infer_end_id'] = resd0['infer_end_id']
    resd['score']=score
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
    print('CAUTION:data has moved to classify_ownership_text_crf.py')
    mydata=[
        {'fname':'form14A_cik1850787_asof20230426_0001213900-23-032897.txt',
         'start_id': 2011, 'end_id': 2567
         },
        {'fname':'form14A_cik6176_asof20230414_0000950170-23-012799.txt',
         'start_id':5163,'end_id':5954
         },
        {'fname':'form14A_cik1508655_asof20230413_0001193125-23-100535.txt',
         'start_id': 427, 'end_id': 716
         },
        {'fname':'form14A_cik1759824_asof20230427_0001193125-23-122351.txt',
         'start_id': 2992, 'end_id': 3230},# on this one content table is wrong
        {'fname':'form14A_cik1644378_asof20240117_0001104659-24-004461.txt',
         'start_id':5625,'end_id':6629
         },# on this one as well content table is wrong because next section is still the ownership
        {'fname':'form14A_cik1570585_asof20230428_0001193125-23-124587.txt',
         'start_id': 1642, 'end_id': 3934}, # another column based content table
        {'fname': 'form14A_cik1598665_asof20230428_0001193125-23-126999.txt',
         'start_id': 2661, 'end_id': 2984},
        {'fname':'form14A_cik1477333_asof20230420_0001477333-23-000033.txt',
        'start_id':7907,'end_id':8489,
         },
        {'fname':'form14A_cik1620280_asof20230413_0001104659-23-044835.txt',
         'start_id': 5002, 'end_id': 5348},
        {'fname':'form14A_cik1598981_asof20230512_0001493152-23-016660.txt',
         'start_id': 6481, 'end_id': 6903},
        {'fname':'form14A_cik1844862_asof20230410_0001104659-23-043334.txt',
         'start_id':13077,'end_id':13723},
        {'fname':'form14A_cik1878313_asof20230419_0001140361-23-019132.txt',
        'start_id':1041,'end_id':1361},
        {'fname':'form14A_cik1661059_asof20230428_0001104659-23-052945.txt',
         'start_id': 5682, 'end_id': 6239},
        {'fname':'form14A_cik1712463_asof20230412_0000950103-23-005629.txt',
         'start_id':4289,'end_id':4547},
        {'fname': 'form14A_cik1606163_asof20230428_0001628280-23-014224.txt',
         'start_id': 4787, 'end_id': 5158},
        {'fname':'form14A_cik1699150_asof20230428_0001140361-23-021647.txt',
         'start_id':13454,'end_id':13824},
        {'fname':'form14A_cik1690080_asof20230523_0001213900-23-042389.txt',
         'start_id':2621,'end_id':2993},
        {'fname':'form14A_cik1782754_asof20240118_0000950170-24-005373.txt',
         'start_id':22991,'end_id':23990},
        {'fname':'form14A_cik19584_asof20230406_0001562762-23-000159.txt',
         'start_id':13494, 'end_id':14524},# this ownership table has a weird format !!!
        {'fname':'form14A_cik1711754_asof20230421_0001213900-23-031675.txt',
        'start_id':3365,'end_id':3641},# has a section : Common Stock Ownership of Directors and Executive Officers that is irrelevant
        {'fname':'form14A_cik1587732_asof20230405_0001193125-23-092215.txt',
         'start_id':9010,'end_id':9222}# multi column content table
    ]
    lscores=[]
    for dataloc in mydata:
        resd = classify_ownership_text(dataloc['fname'])
        score=overlap_score(dataloc['start_id'],dataloc['end_id'],resd['infer_start_id'],resd['infer_end_id'])
        lscores+=[score]



def compute_stats():
    from .edgar_utils import get_forms
    from .blocks_html_v1 import blocks_html
    fnames=get_forms()
    lr=[]
    i=0
    for fname in fnames:
        i+=1
        lstruct, _ = blocks_html(fname)
        resd = classify_ownership_text_from_struct(lstruct)
        resd2 = compare_vs_content_table(fname, resd)
        lr+=[{'fname':fname,'score':resd2['score'],'has_ct':1*(resd2['start_id'] is not None)}]
        if i>600:
            break
    rdf=pd.DataFrame(lr)
    # (Pdb) p rdf[['score','has_ct']].mean()
    # score     0.669518
    # has_ct    0.613861
    # 39   form14A_cik1587732_asof20230405_0001193125-23-092215.txt  0.000000       1



    import pdb;pdb.set_trace()

def generate_examples():
    from pprint import pprint
    from .classify_ownership_text_crf import mydata_test,mydata
    from .edgar_utils import get_random_form
    exclude_files=[x['fname'] for x in mydata]+[x['fname'] for x in mydata_test]
    fname = get_random_form(not_in=exclude_files)
    from .edgar_utils import convert_to_sec_url
    url = convert_to_sec_url(fname)
    resd = classify_ownership_text(fname)
    resd2 = compare_vs_content_table(fname, resd)

    print({k: v for k, v in resd2.items() if k in ['href_ownership_section', 'href_ownership_section_post']})
    print({k: v for k, v in resd.items() if k in ['infer_start_id', 'infer_end_id']})
    print({k: v for k, v in resd2.items() if k in ['start_id', 'end_id']})
    print({k: v for k, v in resd2.items() if k in ['score']})
    print({'fname':fname,'start_id':resd2['start_id'],'end_id':resd2['end_id']})
    # This is for serious debuggin
    if False:
        resd['hmm_path'][['res', 'class']].plot(alpha=0.5, secondary_y='class')
        plt.show()
    #print(resd['hmm_path'].sort_values('res').tail(20))

    hmm_path = resd['hmm_path'].reset_index()
    score_list = resd['score_list']
    #import pdb;pdb.set_trace()

# ipython -i -m IdxSEC.classify_ownership_text_hmm_v1
if __name__=='__main__':
    generate_examples()

