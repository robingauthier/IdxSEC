from bs4 import BeautifulSoup
from . import g_edgar_folder
from lxml import etree
from boilerpy3 import extractors




# c est trop brutal....

# ipython -i -m IdxSEC.test_beautifulsoup_v2
if __name__=='__main__':
    from .filesys import find_files_using_glob
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=list(set(lfilesdf['file'].tolist()))
    from .edgar_utils import get_random_form
    fname='form14A_cik1042729_asof20230414_0001437749-23-010228.txt'
    fname=get_random_form()
    with open(g_edgar_folder + fname, 'rt') as f:
        html = f.read()

    clean_1 = BeautifulSoup(html,features="lxml").get_text()
    with open(g_edgar_folder + '../data_supervised/text_beautifulsoup.txt', 'wt') as f:
        f.write(clean_1)

