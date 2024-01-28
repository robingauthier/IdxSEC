import re
import pandas as pd
from lxml import etree
from .table_utils import filter_dimension_dataframe,remove_duplicated_columns,remove_nan_columns

pd.options.display.max_columns = 10000
pd.set_option('display.expand_frame_repr', False)

def fix_table_values(txt):
    """
    we want to replace 75,071(2) by 75,071
    You cannot remove the % systematically -- you want to keep it for the header
    """
    txt = str(txt)
    txt = re.sub(r'[^\x00-\x7F]+', '', txt)  # removes all unicode characters
    txt = txt.replace('\n', '')
    txt = txt.replace('*', '')
    txt = txt.replace('\t', '')
    txt = txt.replace(',', '')
    txt = txt.replace('  ', ' ')
    if txt=='%':
        return txt
    # Handle the case of a number + % we just remove the %
    matchobj = re.search('^(?P<d>(\d|,|\.)*)\s*%$',txt)
    if matchobj is not None:
        txt=matchobj.group('d')
    # handles : 999,222 % (2)
    # as well as 1.04% (4)(5)
    matchobj = re.search('^(?P<d>(\d|,|\.)*)\s*%*\s*(\(\d*\))*$',txt)
    if matchobj is not None:
        return matchobj.group('d')
    # handles : 999,222 (2)
    matchobj = re.search('^(?P<d>(\d|,|\.)*)\s*(\(\d+\))*$', txt)
    if matchobj is not None:
        return matchobj.group('d')
    # Case
    matchobj = re.search('^(?P<d>(\d|,|\.)*)\s*\([a-z]\)$', txt)
    if matchobj is not None:
        return matchobj.group('d')
    matchobj = re.search('^(?P<d>.*)\s*\(\d+\)$',txt)
    if matchobj is not None:
        return matchobj.group('d')
    # Case FMRLLC(2)245SummerStreet Boston MA02210 we just want FMRLLC
    matchobj = re.search('^(?P<d>.*)\s*\(\d+\).*$',txt)
    if matchobj is not None:
        return matchobj.group('d')
    # Now Cathay pacific reports like this 8,882,814 2/ where 2/ refers to the explanations below
    matchobj = re.search('^(?P<d>(\d|,|\.)*)\s*\d+\/\s*$', txt)
    if matchobj is not None:
        return matchobj.group('d')
    return txt

def convert_html_table_text():
    pass

def test_run():
    from . import g_edgar_folder
    from .edgar_utils import convert_to_sec_url
    from lxml import etree

    fname = 'form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    url=convert_to_sec_url(fname)
    print(fname)
    print('Pls check by hand on :')
    print(url)

    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()

    restree = etree.HTML(ff)

    for elem in restree.xpath('//table'):
        htmlloc = etree.tostring(elem, method='html', encoding=str)
        txtloc = etree.tostring(elem, method='text', encoding=str)
        try:
            l_df_pandas = pd.read_html(htmlloc)
        except Exception as e:
            continue

        if not 'The Vanguard Group' in txtloc:
            continue
        if not 'All executive officers and directors as a group' in txtloc:
            continue
        df_pandas=l_df_pandas[0]
        df_pandas1 = filter_dimension_dataframe(df_pandas, max=1)
        df_pandas1 = remove_duplicated_columns(df_pandas1)
        df_pandas1 = remove_nan_columns(df_pandas1)
        print(df_pandas1)
        import pdb;pdb.set_trace()

    if False:
        nfname=g_edgar_folder + '../data_supervised/text_html_table.txt'
        txt = convert_html_table_text(fname)
        with open(nfname, 'wt') as f:
            f.write(txt)


# ipython -i -m IdxSEC.convert_html_table_text
if __name__=='__main__':
    test_run()
