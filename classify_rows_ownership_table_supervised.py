import pandas as pd
import numpy as np
import re
from .convert_html_table_dataframe import convert_html_table_dataframe

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('expand_frame_repr', False)



def footnote_model(dfloc):
    # quick guess on the footnote
    #fcol=dfloc.iloc[:,0]
    extracted_parts = []
    for id,row in dfloc.iterrows():
        #match=re.search(r'(\(\d*\))', item)
        txt=' '.join([str(v) for k,v in row.items()])
        #if '(1,2,3,6)' in txt:
        #    import pdb;pdb.set_trace()
        match = re.search(r'\(.{1,10}\)', txt)
        if match is None:
            extracted_parts+=['']
        else:
            extracted_parts+=[match.group(0)]
    return extracted_parts

def rowtype_model_v1(dfloc):
    # quick guess on the header/main
    avg_numeric_info=(dflocm == '[N]').mean(axis=1).to_frame('e')
    from sklearn.mixture import GaussianMixture
    lr=[]
    for n_components in [1,2]:
        gaussianmixture = GaussianMixture(n_components=n_components)
        gaussianmixture.fit(avg_numeric_info)
        aic = gaussianmixture.aic(avg_numeric_info)
        lr+=[{'n_components':n_components,'aic':aic,'model':gaussianmixture}]
    rdf=pd.DataFrame(lr)
    infered_n_mixtures=rdf.set_index('n_components')['aic'].idxmin()
    if infered_n_mixtures != 2:
        res = np.array(['M' for x in range(dfloc.shape[0])])
        return res.tolist()
    infered_model = rdf.set_index('n_components').loc[infered_n_mixtures,'model']
    classes_model = infered_model.predict(avg_numeric_info)
    lowest_mean_class = np.argmin(infered_model.means_.flatten())
    res = np.array(['M' for x in range(dfloc.shape[0])])
    res=np.where(classes_model==lowest_mean_class,'H',res)
    return res.tolist()

def rowtype_model_v2(dflocm,debug=0):
    """
    I modelize the number of numerical info vs text+numeric of each row
    by a gaussian mixture that has some strong autocorrelation.
    Hence an HMM where I force to start in the state mean=0
    and I force the transition probabilities to be very low
    """
    from hmmlearn import hmm

    avg_numeric_info = (dflocm == '[N]').sum(axis=1).to_frame('e')
    avg_txt_info = (dflocm == '[T]').sum(axis=1).to_frame('e')
    input_vector = avg_numeric_info/(avg_numeric_info+avg_txt_info)
    input_vector = input_vector.fillna(0.0)

    n_components = 2
    #covariance_type = "tied"
    #covariance_type='diag'
    covariance_type = 'full'
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type,
                            #init_params="cm", params="cm",
                            #init_params="c", params="c",
                            init_params="", params="",
                            n_iter=100)
    #model.startprob_ = np.array([0.5, 0.5])
    model.startprob_ = np.array([1.0, 0.0])
    ac = 0.995
    acm = 1-ac
    model.transmat_=np.array([[ac, acm],[0, 1.0]])
    m1=float(input_vector.mean().iloc[0])
    #print(input_vector)
    #print(m1)
    #import pdb;pdb.set_trace()
    model.means_ = np.array([[0.0],[m1]])
    cov=0.01
    model.covars_=np.array([[[cov]],[[cov]]])
    #model.covars_ = np.array([[cov,0.1], [0.1,cov]])# diag
    #model.covars_ = np.array([[[cov, 0.1], [0.1, cov]])
    #model.fit(input_vector)
    classes_model = model.predict(input_vector)
    lowest_mean_class = np.argmin(model.means_.flatten())
    res = np.array(['M' for x in range(dflocm.shape[0])])
    res=np.where(classes_model==lowest_mean_class,'H',res)

    if debug>=0:
        print('Input to the HMM for header classification :')
        print(input_vector)
        print(classes_model)
        #import pdb;pdb.set_trace()

    return res.tolist()

def get_header(dfloc,dflocm,rowtype):
    # creating the header:
    h_dfloc = dfloc[pd.Series(rowtype)=='H'].astype(str)
    header = (h_dfloc + ' ').sum(axis=0)

    from .text_utils import remove_duplicated_tags
    from .text_utils import clean_txt
    header = header.apply(lambda x:remove_duplicated_tags(x))
    header = header.apply(lambda x: clean_txt(x))

    c_dfloc=dfloc[pd.Series(rowtype)!='H']
    c_dflocm = dflocm[pd.Series(rowtype) != 'H']
    return header,c_dfloc,c_dflocm

def get_header_classes(header,dfm):
    """dfm::the meta type of what is in the table can help guess
    the header name"""
    lr=[]
    for id,elem in header.items():
        s_name=0
        s_share=0
        s_pct=0
        s_other =0
        s_class_a=0
        s_class_b = 0
        s_null = 0
        s_one=1
        elem_lower=elem.lower()
        meta_type=dfm.loc[:,id].value_counts().idxmax()
        if meta_type=='[E]':
            s_other+=4
        if meta_type=='[T]':
            s_other+=1
            s_name+=1
        if meta_type == '[N]':
            s_share+=1
            s_pct+=1
        #import pdb;pdb.set_trace()
        if elem=='':
            s_null+=1
        if 'name' in elem_lower:
            s_name +=1
        if id<=1:
            s_name+=0.4
        if id<=3:
            s_share+=0.2
        if 'share' in elem_lower:
            s_share+=1
        if 'note' in elem_lower: # usually ref to footnote
            s_other += 1
        if 'percent' in elem_lower:
            s_pct+=4
        if '%' in elem_lower:
            s_pct+=4
        if 'days' in elem_lower:
            s_other+=1
        if 'exercisable' in elem_lower:
            s_other+=1
        if 'rights' in elem_lower:
            s_other += 1
        if 'acquire' in elem_lower:
            s_other += 1
        if 'class a' in elem_lower:
            s_class_a+=1
        if 'class b' in elem_lower:
            s_class_b+=1
        lr+=[{'s_name':s_name,'s_share':s_share,'s_pct':s_pct,
              's_other':s_other,'s_classA':s_class_a,'s_classB':s_class_b,
              's_null':s_null,'s_one':s_one}]
    rdf= pd.DataFrame(lr)

    # there can be only 1 column of each

    rdf['pred']=rdf[['s_name', 's_share', 's_pct', 's_other','s_null']].idxmax(axis=1).str.replace('s_','')
    #rdf['predScore'] = rdf[['s_name', 's_share', 's_pct', 's_other', 's_null']].max(axis=1)
    rdf['predClass'] = rdf[['s_classA', 's_classB','s_one']].idxmax(axis=1).str.replace('s_', '')
    #rdf['pred'] = rdf['pred'].where(~rdf['pred'].duplicated(keep='first') | (rdf['pred'] == 'null')|(rdf['pred'] == 'other'), 'null')
    rdf['predClass']=rdf['predClass'].str.replace('one','')
    rdf['pred'] = rdf['pred'].str.replace('null', '')
    #import pdb;pdb.set_trace()
    return rdf

mydata=[
    {
    'fname':'form14A_cik1626971_asof20230428_0001558370-23-007172.txt',
     'html':'<table><tbody><tr><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Shares&nbsp;of&nbsp;Common&nbsp;Stock&nbsp;Beneficial&nbsp;Ownership</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><b>Securities</b></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><b>Number&nbsp;of</b></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Exercisable</b></p></td><td><p><font>​</font></p></td><td><p><b>Shares</b></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Within</b></p></td><td><p><font>​</font></p></td><td><p><b>Beneficially</b></p></td><td><p><font>​</font></p></td><td><p><b>Percent&nbsp;of</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><b>Name of Beneficial Owner</b></p></td><td><p><font>​</font></p></td><td><p><b>Common&nbsp;Stock</b></p></td><td><p><font>​</font></p></td><td><p><b>60&nbsp;Days</b></p></td><td><p><font>​</font></p></td><td><p><b>Owned</b></p></td><td><p><font>​</font></p></td><td><p><b>Total</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><b>5% and Greater Stockholders</b></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td></tr><tr><td><p>OrbiMed Private Investments V,&nbsp;LP(1)</p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p>14.9</p></td><td><p>%</p></td></tr><tr><td><p>Entities affiliated with Adams Street Partners(2)</p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p>7.0</p></td><td><p>%</p></td></tr><tr><td><p>CHI Advisors LLC(3)</p></td><td><p>&nbsp;</p></td><td><p><font> 2,384,402</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 2,384,402</font></p></td><td><p>&nbsp;</p></td><td><p>5.1</p></td><td><p>%</p></td></tr><tr><td><p><b>Named Executive Officers and Directors</b></p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p>&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Richard A. Miller, M.D.(4)</p></td><td><p>&nbsp;</p></td><td><p><font> 1,490,119</font></p></td><td><p>&nbsp;</p></td><td><p><font> 1,717,225</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,207,344</font></p></td><td><p>&nbsp;</p></td><td><p>6.6</p></td><td><p>%</p></td></tr><tr><td><p>Ian T. Clark(5)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Elisha P. (Terry) Gould III(6)</p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,421,866</font></p></td><td><p>&nbsp;</p></td><td><p>7.3</p></td><td><p>%</p></td></tr><tr><td><p>Linda S. Grais, M.D. J.D.(7)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 116,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 116,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Edith P. Mitchell, M.D.(8)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 61,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 61,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Scott W. Morrison(9)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Peter A. Thompson, M.D.(10)</p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 7,089,904</font></p></td><td><p>&nbsp;</p></td><td><p>15.2</p></td><td><p>%</p></td></tr><tr><td><p>Leiv Lea(11)</p></td><td><p>&nbsp;</p></td><td><p><font> 282,444</font></p></td><td><p>&nbsp;</p></td><td><p><font> 555,001</font></p></td><td><p>&nbsp;</p></td><td><p><font> 837,445</font></p></td><td><p>&nbsp;</p></td><td><p>1.8</p></td><td><p>%</p></td></tr><tr><td><p>William B. Jones, Ph.D.(12)</p></td><td><p>&nbsp;</p></td><td><p><font> 133,773</font></p></td><td><p>&nbsp;</p></td><td><p><font> 555,001</font></p></td><td><p>&nbsp;</p></td><td><p><font> 688,774</font></p></td><td><p>&nbsp;</p></td><td><p>1.5</p></td><td><p>%</p></td></tr><tr><td><p>All executive officers and directors as a group (9&nbsp;persons)(13)</p></td><td><p>&nbsp;</p></td><td><p><font> 12,125,606</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,589,727</font></p></td><td><p>&nbsp;</p></td><td><p><font> 15,715,333</font></p></td><td><p>&nbsp;</p></td><td><p>31.3</p></td><td><p>%</p></td></tr></tbody></table>',
     'isOwnership':1,
     'header': ['name', '', 'share', '', 'other', '', '', '', 'pct', ''],
     'headerClass': ['', '', '', '', '', '', '', '', '', ''],
     'footnote':['','','','','','','','(1)','(2)','(3)','','(4)','(5)','(6)','(7)','(8)','(9)','(10)','(11)','(12)','(13)'],
     'rowtype':['H','H','H','H','H','H','H','M','M','M','M','M','M','M','M','M','M','M','M','M','M'],
     },
    {
    'fname':'form14A_cik1524025_asof20230417_0001628280-23-011813.txt',
    'html':'<table><tbody><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td><span>Shares of<span> Class A</span><span> Common</span><span> Stock</span><span> (1)</span></span></td><td></td><td><span>Rights to<span> Acquire</span><span> Class A</span><span> Common</span><span> Stock</span><span> (2)</span></span></td><td></td><td><span>Class A<span> Percentage</span></span></td><td></td><td><span>Shares of<span> Class B</span><span> Common&nbsp;Stock</span></span></td><td></td><td><span>Class B<span> Percentage</span></span></td><td></td><td><span>Total Percentage&nbsp;of<span> Outstanding</span><span> Vote</span></span></td></tr><tr><td><span>Non-Employee Directors and Nominees:</span></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><span>Teresa Aragones</span></td><td><span>10,582&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><span>Erin Chin</span></td><td><span>10,582&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><div><span>Doug Collier </span><span>(3)</span></div></td><td><span>88,776&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><span>Seth Johnson</span></td><td><span>82,476&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><div><span>Janet Kerr </span><span>(4)</span></div></td><td><span>39,486&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><div><span>Bernard Zeichner </span><span>(5)</span></div></td><td><span>47,476&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><span>Named Executive Officers</span></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><div><span>Hezy Shaked </span><span>(6)</span></div></td><td><span>103,000&nbsp;</span></td><td></td><td></td><td><span>125,000&nbsp;</span></td><td></td><td></td><td><span>1.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>7,306,108&nbsp;</span></td><td></td><td></td><td><span>100.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>76.6&nbsp;</span></td><td><span>%</span></td></tr><tr><td><span>Edmond Thomas</span></td><td><span>6,000&nbsp;</span></td><td></td><td></td><td><span>395,350&nbsp;</span></td><td></td><td></td><td><span>1.8&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><span>Michael L. Henry</span></td><td><span>36,600&nbsp;</span></td><td></td><td></td><td><span>72,812&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>*</span></td></tr><tr><td><span>Jonathon D. Kosoff</span></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>31,250&nbsp;</span></td><td></td><td></td><td><span>*</span></td><td></td><td><span>—</span></td><td></td><td><span>—</span></td><td></td><td><span>*</span></td></tr><tr><td><span>All&nbsp;current&nbsp;directors&nbsp;and&nbsp;executive&nbsp;officers as a group (10&nbsp;persons consisting of those named above)</span></td><td><span>424,978</span><span>&nbsp;</span></td><td></td><td></td><td><span>624,412</span><span>&nbsp;</span></td><td></td><td></td><td><span>4.6</span><span>&nbsp;</span></td><td><span>%</span></td><td></td><td><span>7,306,108</span><span>&nbsp;</span></td><td></td><td></td><td><span>100.0</span><span>&nbsp;</span></td><td><span>%</span></td><td></td><td><span>77.5</span><span>&nbsp;</span></td><td><span>%</span></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><span>&gt; 5% Stockholders:</span></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><div><span>Hezy Shaked Living Trust </span><span>(6)</span></div></td><td><span>103,000&nbsp;</span></td><td></td><td></td><td><span>125,000&nbsp;</span></td><td></td><td></td><td><span>1.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>6,212,073&nbsp;</span></td><td></td><td></td><td><span>85.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>65.2&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>Tilly Levine Separate Property Trust </span><span>(7)</span></div></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>1,094,035&nbsp;</span></td><td></td><td></td><td><span>15.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>11.4&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>Fund 1 Investments LLC </span><span>(8)</span></div></td><td><span>5,164,352&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>22.9&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>5.4&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>Divisar Capital Management LLC </span><span>(9)</span></div></td><td><span>1,574,521&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>7.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.6&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>BlackRock, Inc. </span><span>(10)</span></div></td><td><span>1,521,207&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>6.7&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.6&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>Dimensional Fund Advisors LP </span><span>(11)</span></div></td><td><span>1,448,172&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>6.4&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.5&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>The Vanguard Group, Inc. </span><span>(12) </span></div></td><td><span>1,201,906&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>5.3&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.3&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>BML Investment Partners, L.P. </span><span>(13)</span></div></td><td><span>1,154,844&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>5.1&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.2&nbsp;</span></td><td><span>%</span></td></tr><tr><td><div><span>Shay Capital LLC</span><span> (14)</span></div></td><td><span>1,129,862&nbsp;</span></td><td></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>5.0&nbsp;</span></td><td><span>%</span></td><td></td><td><span>—&nbsp;</span></td><td></td><td></td><td><span>—</span></td><td></td><td><span>1.2&nbsp;</span></td><td><span>%</span></td></tr></tbody></table>',
    'isOwnership':1,
    'footnote': ['', '', '', '', '', '(3)', '', '(4)', '(5)', '', '(6)', '', '', '', '', '', '', '(6)', '(7)','(8)', '(9)', '(10)', '(11)', '(12)', '(13)', '(14)'],
    'rowtype': ['H', 'H', 'H', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M','M', 'M', 'M', 'M', 'M', 'M'],
    'header': ['name', 'share', '', 'other', '', 'pct', '', 'share', '', 'pct', '', 'pct', '', '', '', '', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    'headerClass': ['', 'classA', '', 'classA', '', 'classA', '', 'classB', '', 'classB', '', '', '', '', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    },
    {
        #todo:gros probleme d alignement du header
        'fname':'form14A_cik1819790_asof20230426_0001819790-23-000014.txt',
        'html':'<table>\n  <tbody><tr>\n    <td><font>NAME\n    OF BENEFICIAL OWNER</font></td><td><font>&nbsp;</font></td><td><font>AMOUNT\n    AND NATURE OF BENEFICIAL OWNERSHIP</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>PERCENT\n    OF COMMON STOCK OUTSTANDING</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font><b>DIRECTORS,\n    NAMED EXECUTIVE OFFICERS AND 5% STOCKHOLDERS<sup>(1)</sup></b></font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Tony DiMatteo(2)</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>1,489,484</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>2.9</font></td><td><font>%</font></td></tr><tr>\n    <td><font>Matt Clemenson(3)</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>6,289,487</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>12.4</font></td><td><font>%</font></td></tr><tr>\n    <td><font>Ryan Dickinson</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>2,339,286</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>4.6</font></td><td><font>%</font></td></tr><tr>\n    <td><font>Mark Gustavson</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Barney\n    Battles</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Matthew\n    McGahan</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Nick Kounoupias</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Suhail\n    Quraeshi</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Edward\n    Moffly</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>—</font></td><td><font>&nbsp;</font></td></tr><tr>\n    <td><font>Woodford\n    Eurasia Assets Ltd.</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>10,118,257</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>19.9</font></td><td><font>%</font></td></tr><tr>\n    <td><font>DIRECTORS AND EXECUTIVE OFFICERS\n    AS A GROUP (FOUR PERSONS)</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>0</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>&nbsp;</font></td><td><font>0</font></td><td><font>%</font></td></tr></tbody></table>',
        'isOwnership': 1,
    },
    {
        # footnote ref format is (1,2,3,6)
        'fname':'form14A_cik1814974_asof20230420_0001387131-23-004897.txt',
'html':'<table>\n  <tbody><tr>\n    <td><font><b>Name of Beneficial Owner</b></font></td><td>&nbsp;</td><td><font><b>Notes</b></font></td><td>&nbsp;</td><td><font><b>Class A <span> \nCommon Stock Ownership</span></b></font></td><td>&nbsp;</td><td><font><b>Class B <span> \nCommon Stock Ownership</span></b></font></td><td>&nbsp;</td><td><font><b>% of <span> \nClass A </span><span> \nCommon Stock</span></b></font></td><td>&nbsp;</td><td><font><b>% of <span> \nClass B </span><span> \nCommon Stock</span></b></font></td></tr><tr>\n    <td><font>Levan BFC Stock Partners LP</font></td><td>&nbsp;</td><td><font>(1,2,3,6)</font></td><td>&nbsp;</td><td><font>—\n    </font></td><td>&nbsp;</td><td><font>336,915</font></td><td>&nbsp;</td><td><font>2.9%</font></td><td>&nbsp;</td><td><font>8.7%</font></td></tr><tr>\n    <td><font>Levan Partners LLC</font></td><td>&nbsp;</td><td><font>(1,2,3,6)</font></td><td>&nbsp;</td><td><font>986,197</font></td><td>&nbsp;</td><td><font>141,577</font></td><td>&nbsp;</td><td><font>9.8%</font></td><td>&nbsp;</td><td><font>3.7%</font></td></tr><tr>\n    <td><font>Alan B. Levan</font></td><td>&nbsp;</td><td><font>(1,2,3,4,5,6,7)</font></td><td>&nbsp;</td><td><font>1,895,416</font></td><td>&nbsp;</td><td><font>3,710,015</font></td><td>&nbsp;</td><td><font>37.0%</font></td><td>&nbsp;</td><td><font>96.1%</font></td></tr><tr>\n    <td><font>John E. Abdo</font></td><td>&nbsp;</td><td><font>(1,2,3,5)</font></td><td>&nbsp;</td><td><font>1,222,735</font></td><td>&nbsp;</td><td><font>1,495,311</font></td><td>&nbsp;</td><td><font>21.0%</font></td><td>&nbsp;</td><td><font>38.7%</font></td></tr><tr>\n    <td><font>Jarett S. Levan</font></td><td>&nbsp;</td><td><font>(1,2,3,6,7)</font></td><td>&nbsp;</td><td><font>424,622</font></td><td>&nbsp;</td><td><font>536,388</font></td><td>&nbsp;</td><td><font>10.5%</font></td><td>&nbsp;</td><td><font>22.6%</font></td></tr><tr>\n    <td><font>Seth M. Wise</font></td><td>&nbsp;</td><td><font>(1,2,3,7,8)</font></td><td>&nbsp;</td><td><font>474,064</font></td><td>&nbsp;</td><td><font>335,158</font></td><td>&nbsp;</td><td><font>6.9%</font></td><td>&nbsp;</td><td><font>8.7%</font></td></tr><tr>\n    <td><font>Marcia Barry-Smith</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0.0%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Norman H. Becker</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>1,204</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>*</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Andrew R. Cagnetta, Jr.</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>1,000</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>*</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Steven M. Coldren</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>1,893</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>*</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Gregory A. Haile</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0.0%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Willis N. Holcombe</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0.0%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Anthony P. Segreto</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0.0%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Neil Sterling</font></td><td>&nbsp;</td><td><font>(2)</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>0.0%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>Dr. Herbert A. Wertheim</font></td><td>&nbsp;</td><td><font>(1,9)</font></td><td>&nbsp;</td><td><font>793,632</font></td><td>&nbsp;</td><td><font>83,290</font></td><td>&nbsp;</td><td><font>7.6%</font></td><td>&nbsp;</td><td><font>2.2%</font></td></tr><tr>\n    <td><font>Mink Brook Capital GP LLC</font></td><td>&nbsp;</td><td><font>(10)</font></td><td>&nbsp;</td><td><font>813,697</font></td><td>&nbsp;</td><td><font>0</font></td><td>&nbsp;</td><td><font>7.3%</font></td><td>&nbsp;</td><td><font>0.0%</font></td></tr><tr>\n    <td><font>All directors and executive officers of the Company as a group (13 persons)</font></td><td>&nbsp;</td><td><font>(1,2,3,4,5,6,7,8)</font></td><td>&nbsp;</td><td><font>4,082,504</font></td><td>&nbsp;</td><td><font>3,710,015</font></td><td>&nbsp;</td><td><font>51.5%</font></td><td>&nbsp;</td><td><font>96.1%</font></td></tr></tbody></table>',
        'isOwnership': 1,
        'footnote':['','(1,2,3,6)','(1,2,3,6)','','(1,2,3,5)','(1,2,3,6,7)','(1,2,3,7,8)','(2)','(2)','(2)','(2)','(2)','(2)','(2)','(2)','(1,9)','(10)','(13persons)'],
        'rowtype':['H','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'],
        'header':['name','other','other','other','share','other','share','other','pct','other','pct'],
        'headerClass':['','','','','classA','','classB','','classA','','classB'],
},
    {
        'fname':'form14A_cik1819974_asof20230425_0001140361-23-020146.txt',
        'html':'<table><tbody><tr><td><div>Name of Beneficial Owner</div></td><td>​</td><td>​</td><td><div>Shares Beneficially Owned<sup>(1)</sup></div></td></tr><tr><td>​</td><td><div>Shares</div></td><td>​</td><td>​</td><td><div>%</div></td></tr><tr><td><div>Directors and Named Executive Officers:<span> </span></div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td></tr><tr><td><div>Thomas Sonderman<sup>(2)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>646,272</font></div></td><td>​</td><td>​</td><td><div><font>1.45%</font></div></td></tr><tr><td><div>Nancy Fares<sup>(3)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>18,578</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>Gregory B. Graves<sup>(3)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>17,219</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>John T. Kurtzweil<sup>(3)(4)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>22,090</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>Chunyi (Amy) Leong<sup>(3)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>18,578</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>Thomas R. Lujan<sup>(3)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>461,198</font></div></td><td>​</td><td>​</td><td><div><font>1.04%</font></div></td></tr><tr><td><div>Gary J. Obermiller<sup>(3)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>466,769</font></div></td><td>​</td><td>​</td><td><div><font>1.05%</font></div></td></tr><tr><td><div>Loren A. Unterseher<sup>(5)(6)</sup><font></font></div></td><td>​</td><td>​</td><td><div>19,782,379</div></td><td>​</td><td>​</td><td><div>44.56%</div></td></tr><tr><td><div>Steve Manko<sup>(7)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>230,971</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>Bradley Ferguson<sup>(8)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>376,051</font></div></td><td>​</td><td>​</td><td><div><font>*</font></div></td></tr><tr><td><div>All current directors and executive officers as a group (15 persons)<sup>(10)</sup><font></font></div></td><td>​</td><td>​</td><td><div>22,353,058</div></td><td>​</td><td>​</td><td><div>50.34%</div></td></tr><tr><td><div>&nbsp;</div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td></tr><tr><td><div>5% Stockholders:<span> </span></div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td><td>​</td><td>​</td><td><div>&nbsp;</div></td></tr><tr><td><div>CMI Oxbow Partners, LLC<sup>(5)</sup><font></font></div></td><td>​</td><td>​</td><td><div>19,760,289</div></td><td>​</td><td>​</td><td><div>44.52%</div></td></tr><tr><td><div>DDK Developments, LLC<sup>(10)</sup><font></font></div></td><td>​</td><td>​</td><td><div><font>4,624,540</font></div></td><td>​</td><td>​</td><td><div>10.42%</div></td></tr></tbody></table>',

    }

]

def gen_examples():
    from . import g_edgar_folder

    from .edgar_utils import get_random_form
    from .edgar_utils import convert_to_sec_url
    fname=get_random_form()
    #fname='form14A_cik1673481_asof20230627_0001493152-23-022554.txt'
    url=convert_to_sec_url(fname)

    from .extract_structure_form_v2 import filter_html
    nhtml=filter_html(fname,fromid=110,toid=1000000)

    nfname=g_edgar_folder+'../data_supervised/html.html'
    with open(nfname, 'w') as outfile:
        outfile.write(nhtml)
    print('Pls read '+nfname)

def main():
    dataloc = mydata[-1]
    testdf = pd.read_html(dataloc['html'])
    print(testdf[0])
    # import pdb;pdb.set_trace()
    dfloc = convert_html_table_dataframe(dataloc['html'])
    dflocm = convert_html_table_dataframe(dataloc['html'], meta=True)
    print(dflocm)
    if 'rowtype' in dataloc.keys():
        if dfloc.shape[0] != len(dataloc['rowtype']):
            print('Caution you have an issue on nb of rows')

    rowtype = rowtype_model_v2(dflocm)
    footnote = footnote_model(dfloc)
    rdf = pd.DataFrame({'footnote': footnote, 'rowtype': rowtype})
    rdf['fcol'] = dfloc.iloc[:, 0]

    if 'rowtype' in dataloc.keys():
        rowtype = dataloc['rowtype']

    # c_dfloc means content of dfloc
    header, c_dfloc, c_dflocm = get_header(dfloc, dflocm, rowtype)
    assert c_dfloc.shape[0]>0,'issue header is whole df'

    # each header is categorized
    class_header = get_header_classes(header, c_dflocm)
    class_header['header'] = header.str[:50]

    from .table_utils import remove_empty_columns
    from .table_utils import remove_nan_columns
    from .table_utils import cut_content_for_display
    dfloc_c = dfloc.copy()
    dfloc_c = remove_nan_columns(dfloc_c)
    dfloc_c = remove_empty_columns(dfloc_c)
    dfloc_c = cut_content_for_display(dfloc_c, n_cut=20)

    print('Cut table : ')
    print(dfloc_c)
    print('-' * 10)
    print('-' * 10)
    print('Row classification :')
    print(rdf)
    print('-' * 10)
    print('-' * 10)
    print('Column classification :')
    print(class_header)
    print('-' * 10)
    print('-' * 10)
    print('Suggestion : ')
    print('\'footnote\':' + str(rdf['footnote'].tolist()).replace(' ', '') + ',')
    print('\'rowtype\':' + str(rdf['rowtype'].tolist()).replace(' ', '') + ',')
    print('\'header\':' + str(class_header['pred'].tolist()).replace(' ', '') + ',')
    print('\'headerClass\':' + str(class_header['predClass'].tolist()).replace(' ', '') + ',')


# ipython -i -m IdxSEC.classify_rows_ownership_table_supervised
if __name__=='__main__':
    #gen_examples()
    main()
