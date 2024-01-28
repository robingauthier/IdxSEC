import pandas as pd
import numpy as np
from pprint import pprint
import re
from . import g_edgar_folder
import matplotlib
import matplotlib.pyplot as plt
from .classify_ownership_text_hmm_v1 import overlap_score
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn import metrics as skmetrics
#### CRF !!!!!!


lpos=['restricted stock unit',
      ' issued',#shares issued
    'RSU',# Restricted Stock Units
    'SARs',#Stock Appreciation Rights (SARs) like free Stock Options
    'vanguard','blackrock',' fmr ',
      'section 16(a) ',# ownership reporting obligation
    'principal shareholder',
    'security own',#'security ownership'
    'beneficial',
    #'beneficial own',#'beneficial ownership',
    #'beneficially own',
    'security ownership of certain beneficial owners and management', # we see it oftens
    '5%','five percent',
    'voting power',
    'voting right',
    'right to vote',
    'dispositive power',
    #'outstanding share',
    #'share outstanding',
    #'shares outstanding',
    #'outstanding voting share',
    'all executive officers and directors as a group',
    #'shares held ',
    ' within 60 days',
    ' held ',
    ' 13g',
    ' 13d',
    'jointly owning '
    ]
lneu=['(i)','address','shares ','trust']
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

    {'fname':'form14A_cik19584_asof20230406_0001562762-23-000159.txt',
     'start_id':13494, 'end_id':14524},# this ownership table has a weird format !!!
    {'fname':'form14A_cik1711754_asof20230421_0001213900-23-031675.txt',
    'start_id':3365,'end_id':3641},# has a section : Common Stock Ownership of Directors and Executive Officers that is irrelevant
    {'fname':'form14A_cik1587732_asof20230405_0001193125-23-092215.txt',
     'start_id':9010,'end_id':9222},# multi column content table
    {'fname': 'form14A_cik1689657_asof20230414_0001689657-23-000015.txt',
     'start_id': 4897, 'end_id': 5273},
    {'fname': 'form14A_cik77877_asof20230412_0001558370-23-005783.txt',
     'start_id': 23659, 'end_id': 24426},
    {'fname': 'form14A_cik1635977_asof20230411_0001104659-23-043909.txt',
     'start_id': 0, 'end_id': 0},# no section
{'fname': 'form14A_cik105016_asof20230428_0001193125-23-127158.txt', 'start_id': 3111, 'end_id': 3724},
{'fname': 'form14A_cik101984_asof20230426_0000101984-23-000018.txt', 'start_id': 12929, 'end_id': 13283},
{'fname': 'form14A_cik1721484_asof20230428_0001213900-23-033867.txt', 'start_id': 1523, 'end_id': 2002},
{'fname': 'form14A_cik1770787_asof20230428_0001770787-23-000021.txt', 'start_id': 9362, 'end_id': 10020},
{'fname': 'form14A_cik1907982_asof20230424_0001140361-23-019759.txt', 'start_id': 3638, 'end_id': 4026},
{'fname': 'form14A_cik1757898_asof20230607_0001193125-23-162577.txt', 'start_id': 18907, 'end_id': 20086},# 3 tables
{'fname': 'form14A_cik1759546_asof20230411_0001104659-23-044138.txt', 'start_id': 11396, 'end_id': 12359},
{'fname': 'form14A_cik1609804_asof20230407_0001140361-23-017358.txt', 'start_id': 4236, 'end_id': 4533},
{'fname': 'form14A_cik1823086_asof20230508_0001104659-23-057181.txt', 'start_id': 2367, 'end_id': 2719},
{'fname': 'form14A_cik1854458_asof20230420_0001104659-23-047782.txt', 'start_id': 3373, 'end_id': 3785},
{'fname': 'form14A_cik29002_asof20230413_0000950170-23-012615.txt', 'start_id': 1607, 'end_id': 2438},
{'fname': 'form14A_cik1835512_asof20230413_0000950170-23-012712.txt', 'start_id': 4975, 'end_id': 5739},
{'fname': 'form14A_cik1825248_asof20230424_0001140361-23-019841.txt', 'start_id': 1543, 'end_id': 2164},
{'fname': 'form14A_cik39899_asof20230629_0001193125-23-178601.txt', 'start_id': 17070, 'end_id': 17567},# no content table
{'fname': 'form14A_cik1729149_asof20230428_0001729149-23-000053.txt', 'start_id': 1053, 'end_id': 1402},
#{'fname': 'form14A_cik1450445_asof20230630_0001193125-23-180530.txt', 'start_id': 0, 'end_id': 0},# fund irrelevant. do we include it???
{'fname': 'form14A_cik1607939_asof20230427_0001193125-23-122336.txt', 'start_id': 9095, 'end_id': 9528},
{'fname': 'form14A_cik1784567_asof20230530_0001784567-23-000051.txt', 'start_id': 3841, 'end_id': 4128},
{'fname': 'form14A_cik33002_asof20230530_0000950170-23-024756.txt', 'start_id': 5706, 'end_id': 6269},
{'fname': 'form14A_cik1850838_asof20230425_0000950170-23-014247.txt', 'start_id': 4698, 'end_id': 5003},
{'fname': 'form14A_cik1590750_asof20230428_0001193125-23-124500.txt', 'start_id': 2016, 'end_id': 2372},
{'fname': 'form14A_cik1702780_asof20230427_0001628280-23-013656.txt', 'start_id': 7249, 'end_id': 7879},
{'fname': 'form14A_cik1636282_asof20230417_0001193125-23-103731.txt', 'start_id': 3022, 'end_id': 3471},

    # run ipython -i -m IdxSEC.classify_ownership_text_hmm_v1

]
mydata_test=[
    {'fname': 'form14A_cik1661059_asof20230428_0001104659-23-052945.txt',
     'start_id': 5682, 'end_id': 6239},
    {'fname': 'form14A_cik1712463_asof20230412_0000950103-23-005629.txt',
     'start_id': 4289, 'end_id': 4547},
    {'fname': 'form14A_cik1606163_asof20230428_0001628280-23-014224.txt',
     'start_id': 4787, 'end_id': 5158},
    {'fname': 'form14A_cik1699150_asof20230428_0001140361-23-021647.txt',
     'start_id': 13454, 'end_id': 13824},
    {'fname': 'form14A_cik1690080_asof20230523_0001213900-23-042389.txt',
     'start_id': 2621, 'end_id': 2993},
    {'fname': 'form14A_cik1782754_asof20240118_0000950170-24-005373.txt',
     'start_id': 22991, 'end_id': 23990},
{'fname': 'form14A_cik1607939_asof20230427_0001193125-23-122336.txt', 'start_id': 9095, 'end_id': 9528},
{'fname': 'form14A_cik1784567_asof20230530_0001784567-23-000051.txt', 'start_id': 3841, 'end_id': 4128},
{'fname': 'form14A_cik33002_asof20230530_0000950170-23-024756.txt', 'start_id': 5706, 'end_id': 6269},

]
def percentage_uppercase(input_string):
    total_chars = len(input_string)
    uppercase_chars = sum(1 for char in input_string if char.isupper())
    percentage = (uppercase_chars / total_chars) if total_chars > 0 else 0.0
    return percentage


def compute_features(elem,doclen=100):
    sent=elem['content']
    sent_lower = sent.lower()
    rd={}
    rd['len']=len(sent)/150 if len(sent)<150 else 1
    rd['pctUp']=percentage_uppercase(sent)
    rd['pctUpStart'] = percentage_uppercase(sent[:30])
    rd['tag_table']=1.0*(elem['tag']=='table')
    rd['elemi_pct']=elem['elemi']/doclen
    rd['elemi_start']=1.0*(rd['elemi_pct']<=0.1)
    rd['elemi_end'] = 1.0 * (rd['elemi_pct'] >= 0.9)
    for w in lpos:
        wcnt=sent_lower.count(w.lower())
        rd['w_'+w.replace(' ','_')+'_pos']=wcnt/5 if wcnt<5 else 1.0
    for w in lneg:
        wcnt=sent_lower.count(w.lower())
        rd['w_'+w.replace(' ','_')+'_neg']=wcnt/5 if wcnt<5 else 1.0
    for w in lneu:
        wcnt=sent_lower.count(w.lower())
        rd['w_'+w.replace(' ','_')+'_neu']=wcnt/5 if wcnt<5 else 1.0


    #  'vest','vested','vesting'
    # (?: means non capturing group
    match=re.search(r'\svest(?:ing|ed)?\s', sent_lower)
    if match is not None:
        rd['w_vest']=1.0
    else:
        rd['w_vest'] = 0.0

    match1=re.search(r'shares?\s+outstanding', sent_lower)
    match2=re.search(r'outstanding.{1,15}share', sent_lower)
    if (match1 is not None) or (match2 is not None):
        rd['shout']=1.0
    else:
        rd['shout'] = 0.0

    match=re.search(r'(\(\d* persons\))', sent_lower)
    if match is not None:
        rd['x_person']=1.0
    else:
        rd['x_person'] = 0.0
    #rd['x_person']=rd['x_person']/5.0 if rd['x_person']<=5 else 1.0

    # Anything that talks about shares
    match = re.search(r'[\d,]+\s+share', sent_lower)
    if match is not None:
        rd['x_share']=1.0
    else:
        rd['x_share'] = 0.0
    #rd['x_share'] = rd['x_share'] / 5.0 if rd['x_share'] <= 5 else 1.0

    match = re.search(r'\((?P<id>.{1,5})\)', sent_lower)
    if match is not None:
        rd['mention']=1.0
    else:
        rd['mention'] = 0.0

    match = re.search(r'\((?P<id>\d{1,5})\)', sent_lower)
    if match is not None:
        rd['mention_num']=1.0
    else:
        rd['mention_num'] = 0.0

    return rd

def compute_features_basic(elem,doclen=100):
    sent=elem['content']
    sent_lower = sent.lower()
    rd={}
    rd['len']=len(sent)/150 if len(sent)<150 else 1
    rd['pctUp']=percentage_uppercase(sent)
    rd['pctUpStart'] = percentage_uppercase(sent[:30])
    rd['tag_table']=1.0*(elem['tag']=='table')
    rd['elemi_pct']=elem['elemi']/doclen
    rd['elemi_start']=1.0*(rd['elemi_pct']<=0.1)
    rd['elemi_end'] = 1.0 * (rd['elemi_pct'] >= 0.9)
    from gensim.parsing.preprocessing import preprocess_string
    tmp_tokens = preprocess_string(sent_lower) #stemming
    tmp_tokens = [x for x in tmp_tokens if len(x) <= 15]
    #import pdb;pdb.set_trace()
    for token in tmp_tokens:
        rd['w_'+token]=tmp_tokens.count(token)

    match = re.search(r'\((?P<id>.{1,5})\)', sent_lower)
    if match is not None:
        rd['mention']=1.0
    else:
        rd['mention'] = 0.0
    return rd


def test_compute_features():
    elem={'tag':'div','elemi':0,'content':'(1) XXX has 9,000 shares outstanding within 60 days'}
    rd=compute_features(elem)
    assert rd['x_share']>0,'issue'
    assert rd['mention_num'] > 0, 'issue'
    assert rd['shout'] > 0, 'issue'
    assert rd['tag_table']==0,'issue'
    assert rd['w__within_60_days_pos']>0,'issue'

    elem = {'tag': 'table', 'elemi': 0, 'content': '(i) XXX has 9,000  outstanding voting shares'}
    rd = compute_features(elem)
    assert rd['mention'] > 0, 'issue'
    assert rd['shout'] > 0, 'issue'
    assert rd['tag_table'] > 0, 'issue'

    #import pdb;pdb.set_trace()


def create_train_data(mydata):
    X=[]
    y=[]
    for dataloc in mydata:
        Xdoc=[]
        ydoc=[]
        from .blocks_html_v1 import blocks_html
        fname=dataloc['fname']
        start_id=dataloc['start_id']
        end_id=dataloc['end_id']
        lstruct, _ = blocks_html(fname)
        for elem in lstruct:
            elemi=elem['elemi']
            # the basic features seems to enable better predictions than mine...
            Xdoc += [compute_features_basic(elem, doclen=len(lstruct))]
            #Xdoc+=[compute_features(elem,doclen=len(lstruct))]
            if (elemi<start_id) or (elemi>end_id):
                label='other'
            else:
                label='ownership'
            ydoc+=[label]
        X+=[Xdoc]
        y+=[ydoc]
    return X,y

def train_model():
    X_train,y_train=create_train_data(mydata)
    X_test,y_test=create_train_data(mydata_test)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,# l1 reg
        c2=0.1,# l2 reg
        max_iterations=100,
        all_possible_transitions=True
    )
    labels = ['ownership','other']
    print('Training CRF')
    crf.fit(X_train, y_train)
    y_pred=crf.predict(X_test)
    r=metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)
    prec=metrics.flat_precision_score(y_test, y_pred,average='weighted', labels=['ownership'])
    print(skmetrics.classification_report(np.concatenate(y_test), np.concatenate(y_pred)))

    from collections import Counter
    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f | %-8s | %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(60))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-60:])

    from .model_util import save_sklearn_model,load_sklearn_model
    save_sklearn_model(crf, 'crf_ownership_part')
    nmodel=load_sklearn_model('crf_ownership_part')
    import pdb;pdb.set_trace()



# ipython -i -m IdxSEC.classify_ownership_text_crf
if __name__=='__main__':
    train_model()
    #test_compute_features()

