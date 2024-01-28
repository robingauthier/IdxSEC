import pandas as pd
import numpy as np
import os
from . import g_edgar_folder
from .filesys import append_to_file
from .extract_table_content_v1 import extract_table_content
import distance

g_table_content_training = g_edgar_folder + '../data_supervised/table_content_training.txt'

# how does that work :
# rather than pasting the full content table, I am just providing an href that is in the content table


mydata=[
# previous test_1
{'section': 'Stock Ownership',
 'href': '#i8460927f3ea9449f8b812d7bfe34961b_172',
'isOwnership':1,
 'fname':'form14A_cik898174_asof20230413_0000898174-23-000081.txt'},
{'section': 'Voting',
 'href': '#i8460927f3ea9449f8b812d7bfe34961b_202',
 'fname':'form14A_cik898174_asof20230413_0000898174-23-000081.txt'},
{'section': 'Use of Non-GAAP Financial Measures',
 'href': '#i8460927f3ea9449f8b812d7bfe34961b_214',
 'fname':'form14A_cik898174_asof20230413_0000898174-23-000081.txt'},
# previous test_2 : nothing to find
{'fname':'form14A_cik718877_asof20230501_0001308179-23-000834.txt'},
# previous test_4
{'section':'Effects of the Reverse Stock Split',
 'href':'#effective_of_stock_split',
 'fname':'form14A_cik1816017_asof20230501_0000950170-23-016022.txt'},
{'section': 'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
 'href': '#security_ownership_of_certain_benefical',
'isOwnership':1,
 'fname':'form14A_cik1816017_asof20230501_0000950170-23-016022.txt'},
# previous test_3
{'section': 'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS',
 'href': '#SOOC',
'isOwnership':1,
 'fname': 'form14A_cik1819576_asof20230428_0001104659-23-053124.txt'},
{'section': 'EXECUTIVE OFFICERS AND DIRECTOR AND OFFICER COMPENSATION',
 'href': '#EOAD',
 'fname': 'form14A_cik1819576_asof20230428_0001104659-23-053124.txt'},
{
'fname':'form14A_cik1473844_asof20230425_0001104659-23-049146.txt',
'section':'BENEFICIAL OWNERSHIP OF THE COMPANY’S COMMON STOCK BY MANAGEMENT AND PRINCIPAL SHAREHOLDERS OF THE COMPANY',
'href':'#tBOOT',
'isOwnership':1,
},
{
# first table
'fname': 'form14A_cik1473844_asof20230425_0001104659-23-049146.txt',
'section': 'Compensation Discussion and Analysis',
'href': '#tCDAA',
'isOwnership': 0,
},
{'fname':'form14A_cik1038277_asof20230524_0001515971-23-000078.txt',# only 1 table here
'href':'#10748-09',
'section':'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'isOwnership': 1,
 },
{   # blackrock fund no ownership section relevant
'fname':'form14A_cik1232860_asof20230523_0001193125-23-151823.txt',
'href':'#toc499976_5',
'isOwnership': 0,
'section':'INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM',
},
{
'fname':'form14A_cik1847440_asof20230607_0001104659-23-068918.txt',
'section':'BENEFICIAL OWNERSHIP OF SECURITIES',
'href':'#sp1-046',
'isOwnership':1,
},
{
# only 1 table here easy
'fname':'form14A_cik1722010_asof20230512_0001628280-23-017962.txt',
'section':'BENEFICIAL OWNERSHIP OF COMMON STOCK',
'href':'#i8d5010b6108d4a36b3022faa32106d3e_64',
'isOwnership':1,
},
{
# there are no content tables here
'fname':'form14A_cik1778129_asof20230420_0000950170-23-013690.txt',
'section':'',
'href':'',
'isOwnership':0,
},
{
# blackstone
'fname':'form14A_cik1061630_asof20230427_0001193125-23-122168.txt',
'section':'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'href':'#toc451714_10',
'isOwnership':1,
},
{
'fname':'form14A_cik1810560_asof20230505_0000950170-23-018333.txt',
'section':'',
'href':'',
'isOwnership':1,
},
{
'fname':'form14A_cik1401667_asof20230428_0001437749-23-011500.txt',
'section':'SECURITY OWNERSHIP OF DIRECTORS, AND EXECUTIVE OFFICERS AND CERTAIN BENEFICIAL OWNERS',
'href':'#securityownership',
'isOwnership':1,
},
{
# the table of content is over 2 pages
'fname':'form14A_cik1401667_asof20230428_0001437749-23-011500.txt',
'section':'AUDIT MATTERS',
'href':'#auditmatters',
'isOwnership':0,
},

{
'fname':'form14A_cik1796129_asof20230412_0001193125-23-099195.txt',
'section':'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'href':'#toc381653_19',
'isOwnership':1,
},
{
'fname':'form14A_cik1722438_asof20230426_0001722438-23-000069.txt',
'section':'Ownership of Common Stock by 5% or More Holders',
'href':'#i6d642650d222432093ee238fb982bf95_130',
'isOwnership':1,
},
{
'fname':'form14A_cik1722438_asof20230426_0001722438-23-000069.txt',
'section':'Ownership of Common Stock by Directors and Executive Officers',
'href':'#i6d642650d222432093ee238fb982bf95_133',
'isOwnership':1,
},
{
# table is split on 2
'fname':'form14A_cik1722438_asof20230426_0001722438-23-000069.txt',
'section':'Director Independence',
'href':'#i6d642650d222432093ee238fb982bf95_67',
'isOwnership':0,
},
{
# ISSUE:the title of the section is the page... info is in the table elsewhere
'fname':'form14A_cik1093557_asof20230406_0001093557-23-000099.txt',
'section':'77',
'href':'#i32c36b02d0304f4b81cf207652f2f701_196',
'isOwnership':1,
},
{
# ISSUE:the title of the section is the page... info is in the table elsewhere
'fname':'form14A_cik1093557_asof20230406_0001093557-23-000099.txt',
'section':'1',
'href':'#i32c36b02d0304f4b81cf207652f2f701_31',
'isOwnership':0,
},
{
'fname':'form14A_cik1674365_asof20230602_0001104659-23-067587.txt',
'section':'Security Ownership of Certain Beneficial Owners and Management',
'href':'#tSOOC',
'isOwnership':1,
},
{
'fname':'form14A_cik1658566_asof20230411_0001308179-23-000637.txt',
'section':'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'href':'#new_id-112',
'isOwnership':1,
},
{
# on this form each row of the table of content is a table...
'fname':'form14A_cik1658566_asof20230411_0001308179-23-000637.txt',
'section':'QUESTIONS AND ANSWERS ABOUT THE ANNUAL MEETING',
'href':'#new_id-155',
'isOwnership':0,
},
{
# on this form each row of the table of content is a table...
'fname':'form14A_cik1658566_asof20230411_0001308179-23-000637.txt',
'section':'GENERAL INFORMATION',
'href':'#new_id-150',
'isOwnership':0,
},
{
'fname':'form14A_cik1327607_asof20230427_0001327607-23-000044.txt',
'section':'BENEFICIAL OWNERSHIP OF THE COMPANY’S COMMON STOCK BY MANAGEMENT AND PRINCIPAL SHAREHOLDERS OF THE COMPANY',
'href':'#i8ca046193f764a4b9231478b4a4b7873_229',
'isOwnership':1,
},
{
'fname':'form14A_cik1652044_asof20230421_0001308179-23-000736.txt',
'section':'COMMON STOCK OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'href':'#lgooga018',
'isOwnership':1,
},
{
'fname':'form14A_cik1652044_asof20230421_0001308179-23-000736.txt',
'section':'AUDIT MATTERS',
'href':'#lgooga043b',
'isOwnership':0,
},
{
'fname':'form14A_cik851310_asof20230428_0001140361-23-021538.txt',
'section':'Security Ownership of Certain Beneficial Owners and Management',
'href':'#tSOC',
'isOwnership':1,
},
{
'fname':'form14A_cik1111928_asof20230413_0001111928-23-000071.txt',
'section':'Common Stock Ownership',
'href':'#iedf6ea2e7c4a469ea397579cf6f6ecfe_91',
'isOwnership':1,
},
{
'fname':'form14A_cik1216583_asof20230630_0001193125-23-178922.txt',
'section':'Beneficial Ownership for Tax Purposes',
'href':'#appxb462050_331',
'isOwnership':0,
},
{
'fname':'form14A_cik1216583_asof20230630_0001193125-23-178922.txt',
'section':'Transfers',
'href':'#appxb462050_230',
'isOwnership':0,
},
{
'fname':'form14A_cik1659166_asof20230424_0001193125-23-113067.txt',
'section':'Ownership of Our Stock',
'href':'#txa745814_4',
'isOwnership':1,
},
{
'fname':'form14A_cik1659166_asof20230424_0001193125-23-113067.txt',
'section':'Our Commitment to Sustainability',
'href':'#txa745814_19',
'isOwnership':0,
},
{
'fname':'form14A_cik1626971_asof20230428_0001558370-23-007172.txt',
'section':'SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT',
'href':'#SECURITYOWNERSHIPOFCERTAINBENEFICIALOWNE',
'isOwnership':1,
}
]

mydatadf=pd.DataFrame(mydata)

def create_target_data(reset=True):
    """
    Target data for classifying if it is or not a content table !!!


    the goal is 1 table = 1 row of features
    and we add the target = 0 or 1
    but here we only store the minimum information with the target column

    because later we might add more features
    hence the minimum is
    fname           hash                 target
    form14A_*.txt  -4159038636281539092     0
    form14A_*.txt  2804821545667600245      1

    """
    if reset:
        try:
            os.remove(g_table_content_training)
        except Exception as e:
            pass

    lfiles = list(set(mydatadf['fname'].tolist()))
    for fname in lfiles:
        #if fname=='form14A_cik1473844_asof20230425_0001104659-23-049146.txt':
        #    import pdb;pdb.set_trace()
        #if fname=='form14A_cik1652044_asof20230421_0001308179-23-000736.txt':
        #    import pdb;pdb.set_trace()
        mydatadfloc=mydatadf.loc[lambda x:x['fname']==fname].copy()
        mydatadfloc['target']=1.0
        mydatadfloc=mydatadfloc.rename(columns={'section':'section1'})

        searchlogdf, content_table = extract_table_content(fname,debug=0)
        if content_table is None:
            continue
        # match on href has to be exact
        ncontent_table = content_table.merge(mydatadfloc[['section1','href','target','isOwnership']],on='href',how='left')

        # but match on section can be approximative
        ncontent_table['dst']=1.0
        for id,row in ncontent_table.iterrows():
            if pd.isna(row['section1']):
                continue
            dst=distance.jaccard(row['section'],row['section1'])
            ncontent_table.loc[id,'dst']=dst

        if False:
            ncontent_table['section']=ncontent_table['section'].str[:20]
            ncontent_table['section1'] = ncontent_table['section1'].str[:20]
            print(ncontent_table[['section','section1','dst']])

        ncontent_table['target']=ncontent_table['target'].fillna(0.0)
        ncontent_table['isOwnership'] = ncontent_table['isOwnership'].fillna(0.0)
        ncontent_table['target'] = np.where(ncontent_table['dst']>0.2,0,ncontent_table['target'])

        nres=ncontent_table.groupby(['hash']).agg({'target':'max','isOwnership':'max'}).reset_index()
        searchlogdf=searchlogdf.merge(nres,on=['hash'],how='left')
        searchlogdf['target']=searchlogdf['target'].fillna(0.0)
        append_to_file(searchlogdf, g_table_content_training)


def help_manual_work(fname=None):
    import random
    from .edgar_utils import get_forms
    from .edgar_utils import convert_to_sec_url
    lfiles=get_forms()
    if fname is None:
        fname=random.choice(lfiles)
    if fname in mydatadf['fname'].tolist():
        print('!!!!!!!!!!!!WARNING!!!!!!!!!!!!!')
        print('creating content table supervised data :: you already did this fname')
        print('!!!!!!!!!!!!WARNING!!!!!!!!!!!!!')
    #fname='form14A_cik898174_asof20230413_0000898174-23-000081.txt'
    url = convert_to_sec_url(fname)
    searchlogdf, content_table = extract_table_content(fname)
    if content_table is None:
        return help_manual_work()
    print('Please fill:')
    print('{')
    print('\'fname\':\''+fname+'\',')
    print('\'section\':\'\',')
    print('\'href\':\'\',')
    print('\'isOwnership\':1,')
    print('}')
    print('Pls check for the online version : ')
    print(url)
    print('-'*20)
    print('-' * 20)
    print('-' * 20)
    nfname=g_edgar_folder + '../data_supervised/table_content.txt'
    print('Please open the file to see table of content : '+nfname)
    print(content_table.assign(cnt=1).groupby('hash').agg({'cnt':'sum'}))
    with open(nfname, 'wt') as f:
        if content_table is not None:
            content_table.to_string(f)
        else:
            f.write('No content table found')
    # this bit is not necessary
    from .convert_html_text import convert_html_to_text
    nfname=g_edgar_folder + '../data_supervised/text_html.txt'
    txt = convert_html_to_text(fname)
    with open(nfname, 'wt') as f:
        f.write(txt)


def test_1():
    """we test that the data we create is correct"""
    create_target_data(reset=True)
    df = pd.read_csv(g_table_content_training)
    fname='form14A_cik1038277_asof20230524_0001515971-23-000078.txt'
    dfloc=df[df['fname']==fname]
    print(dfloc)
    assert dfloc.shape[0]==1,'issue'
    assert dfloc['hash'].iloc[0]=='48fff4e917','issue'
    assert dfloc['target'].iloc[0] >0, 'issue'

def test_2():
    """we test that the data we create is correct"""
    create_target_data(reset=True)
    df = pd.read_csv(g_table_content_training)
    fname='form14A_cik1847440_asof20230607_0001104659-23-068918.txt'
    dfloc=df[df['fname']==fname]
    print(dfloc)
    assert dfloc.shape[0]==1,'issue'
    assert dfloc['target'].iloc[0] >0, 'issue'

def test_3():
    create_target_data(reset=True)
    df = pd.read_csv(g_table_content_training)
    fname='form14A_cik1401667_asof20230428_0001437749-23-011500.txt'
    dfloc=df[df['fname']==fname]
    print(dfloc)
    assert dfloc.shape[0] == 2, 'issue'
    assert np.all(dfloc['target']>0), 'issue'
    assert not np.all(dfloc['isOwnership'] > 0), 'issue'

def test_4():
    create_target_data(reset=True)
    df = pd.read_csv(g_table_content_training)
    fname='form14A_cik1093557_asof20230406_0001093557-23-000099.txt'
    dfloc=df[df['fname']==fname]
    print(dfloc)
    assert dfloc.shape[0] == 3, 'issue'
    v1=pd.Series(dfloc['target'].astype(int).values)
    assert v1.equals(pd.Series([1,1,1])), 'issue'
    v1 = pd.Series(dfloc['isOwnership'].astype(int).values)
    assert v1.equals(pd.Series([0, 0, 1])), 'issue'

def test_5():
    create_target_data(reset=True)
    df = pd.read_csv(g_table_content_training)
    fname='form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    dfloc=df[df['fname']==fname].drop_duplicates(subset=['hash'])
    print(dfloc)
    assert dfloc['target'].sum()==2,'issue'
    assert dfloc['isOwnership'].sum() == 1, 'issue'

# ipython -i -m IdxSEC.extract_table_content_supervised
if __name__=='__main__':
    print('Nb of supervised samples : %i'%mydatadf.shape[0])
    #print(mydatadf)
    #test_1()
    #test_2()
    #test_3()
    #test_4()
    #test_5()
    if True:
        # ...                                                        ...
        # form14A_cik1363829_asof20230421_0001363829-23-000132.txt   100
        # form14A_cik1037976_asof20230414_0001308179-23-000664.txt   102
        # form14A_cik1058290_asof20230421_0001308179-23-000721.txt   104
        # form14A_cik1800682_asof20230522_0001193125-23-149583.txt   108
        # form14A_cik1658566_asof20230411_0001308179-23-000637.txt   109
        #                                                           hash
        help_manual_work()
    if False:
        create_target_data(reset=True)
    if False:
        df=pd.read_csv(g_table_content_training)
        print(df.shape[0])
        print(df[df['fname']=='form14A_cik1473844_asof20230425_0001104659-23-049146.txt'])