import pandas as pd
import numpy as np
from lxml import etree
from .text_utils import clean_txt

txt='<table><tbody><tr><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td><td><div><div><p><font>​</font></p></div></div></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Shares&nbsp;of&nbsp;Common&nbsp;Stock&nbsp;Beneficial&nbsp;Ownership</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><b>Securities</b></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><b>Number&nbsp;of</b></p></td><td><p>&nbsp;&nbsp;&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Exercisable</b></p></td><td><p><font>​</font></p></td><td><p><b>Shares</b></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><b>Within</b></p></td><td><p><font>​</font></p></td><td><p><b>Beneficially</b></p></td><td><p><font>​</font></p></td><td><p><b>Percent&nbsp;of</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><b>Name of Beneficial Owner</b></p></td><td><p><font>​</font></p></td><td><p><b>Common&nbsp;Stock</b></p></td><td><p><font>​</font></p></td><td><p><b>60&nbsp;Days</b></p></td><td><p><font>​</font></p></td><td><p><b>Owned</b></p></td><td><p><font>​</font></p></td><td><p><b>Total</b></p></td><td><p>&nbsp;</p></td></tr><tr><td><p><b>5% and Greater Stockholders</b></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td><td><p><font>​</font></p></td></tr><tr><td><p>OrbiMed Private Investments V,&nbsp;LP(1)</p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p>14.9</p></td><td><p>%</p></td></tr><tr><td><p>Entities affiliated with Adams Street Partners(2)</p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p>7.0</p></td><td><p>%</p></td></tr><tr><td><p>CHI Advisors LLC(3)</p></td><td><p>&nbsp;</p></td><td><p><font> 2,384,402</font></p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 2,384,402</font></p></td><td><p>&nbsp;</p></td><td><p>5.1</p></td><td><p>%</p></td></tr><tr><td><p><b>Named Executive Officers and Directors</b></p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p>&nbsp;</p></td><td><p><font>​</font></p></td><td><p>&nbsp;</p></td><td><p>&nbsp;&nbsp;</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Richard A. Miller, M.D.(4)</p></td><td><p>&nbsp;</p></td><td><p><font> 1,490,119</font></p></td><td><p>&nbsp;</p></td><td><p><font> 1,717,225</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,207,344</font></p></td><td><p>&nbsp;</p></td><td><p>6.6</p></td><td><p>%</p></td></tr><tr><td><p>Ian T. Clark(5)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Elisha P. (Terry) Gould III(6)</p></td><td><p>&nbsp;</p></td><td><p><font> 3,275,616</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,421,866</font></p></td><td><p>&nbsp;</p></td><td><p>7.3</p></td><td><p>%</p></td></tr><tr><td><p>Linda S. Grais, M.D. J.D.(7)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 116,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 116,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Edith P. Mitchell, M.D.(8)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 61,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 61,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Scott W. Morrison(9)</p></td><td><p>&nbsp;</p></td><td><p><font> —</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p>*</p></td><td><p><font>​</font></p></td></tr><tr><td><p>Peter A. Thompson, M.D.(10)</p></td><td><p>&nbsp;</p></td><td><p><font> 6,943,654</font></p></td><td><p>&nbsp;</p></td><td><p><font> 146,250</font></p></td><td><p>&nbsp;</p></td><td><p><font> 7,089,904</font></p></td><td><p>&nbsp;</p></td><td><p>15.2</p></td><td><p>%</p></td></tr><tr><td><p>Leiv Lea(11)</p></td><td><p>&nbsp;</p></td><td><p><font> 282,444</font></p></td><td><p>&nbsp;</p></td><td><p><font> 555,001</font></p></td><td><p>&nbsp;</p></td><td><p><font> 837,445</font></p></td><td><p>&nbsp;</p></td><td><p>1.8</p></td><td><p>%</p></td></tr><tr><td><p>William B. Jones, Ph.D.(12)</p></td><td><p>&nbsp;</p></td><td><p><font> 133,773</font></p></td><td><p>&nbsp;</p></td><td><p><font> 555,001</font></p></td><td><p>&nbsp;</p></td><td><p><font> 688,774</font></p></td><td><p>&nbsp;</p></td><td><p>1.5</p></td><td><p>%</p></td></tr><tr><td><p>All executive officers and directors as a group (9&nbsp;persons)(13)</p></td><td><p>&nbsp;</p></td><td><p><font> 12,125,606</font></p></td><td><p>&nbsp;</p></td><td><p><font> 3,589,727</font></p></td><td><p>&nbsp;</p></td><td><p><font> 15,715,333</font></p></td><td><p>&nbsp;</p></td><td><p>31.3</p></td><td><p>%</p></td></tr></tbody></table>'


def convert_html_table_dataframe(txt,meta=False,debug=0):
    restree=etree.HTML(txt)
    assert txt[:6]=='<table','issue convert_html_table_dataframe : input not a table'
    lr=[]
    for rowelem in restree.xpath('//tr'):
        lc = []
        row=etree.HTML(etree.tostring(rowelem,method='html',encoding=str))
        for colelem in row.xpath('//td'):
            txtloch=etree.tostring(colelem, method='html', encoding=str)
            txtloc=etree.tostring(colelem, method='text', encoding=str)
            if '<b>' in txtloch:
                txtloc=txtloc+' [B]'
            txtloc_c = clean_txt(txtloc)
            txtloc_meta = clean_txt(txtloc,meta=meta)
            if meta:
                lc += [txtloc_meta]
            else:
                lc+=[txtloc_c]
        lr+=[lc]
    ncols=np.array([len(x) for x in lr])
    check_colsize=np.all(ncols==np.max(ncols))
    if (debug>0) and (not check_colsize):
        print('Warning there is not the same number of columns in each row')
    df=pd.DataFrame(lr)
    return df.fillna('')

# ipython -i -m IdxSEC.convert_html_table_dataframe
if __name__=='__main__':
    df=convert_html_table_dataframe(txt,meta=False)
    print(df.to_string())
    dfm = convert_html_table_dataframe(txt, meta=True)
    print(dfm.to_string())
    print(df.to_json())

