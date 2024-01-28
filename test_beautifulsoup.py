from bs4 import BeautifulSoup
from . import g_edgar_folder
from lxml import etree
from boilerpy3 import extractors

extractor = extractors.ArticleExtractor()
# https://www.kaggle.com/code/mohammadbolandraftar/nlp-text-pre-processing-how-to-clean-html-format

# ipython -i -m IdxSEC.test_beautifulsoup
if __name__=='__main__':
    from .filesys import find_files_using_glob
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=list(set(lfilesdf['file'].tolist()))

    fname='form14A_cik1042729_asof20230414_0001437749-23-010228.txt'
    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)
    # In [6]: [len(etree.tostring(x,method='html')) for x in restree.xpath('//xbrl')]
    # Out[6]: [1658131, 7864, 10019, 20190, 9875]
    #assert len(restree.xpath('//html'))==1,'issue'
    #html0 = restree.xpath('//html')[0]
    html0 = restree.xpath('//xbrl')[0]
    html = etree.tostring(html0,method='html',encoding=str)
    #import pdb;pdb.set_trace()
    clean_1 = BeautifulSoup(html,features="lxml").get_text()
    with open(g_edgar_folder + '../data_supervised/text_beautifulsoup.txt', 'wt') as f:
        f.write(clean_1)

    clean_2 = etree.tostring(html0, method='text', encoding=str)
    with open(g_edgar_folder + '../data_supervised/text_etree.txt', 'wt') as f:
        f.write(clean_2)

    clean_3 = content = extractor.get_content(html)
    with open(g_edgar_folder + '../data_supervised/text_boilerpy.txt', 'wt') as f:
        f.write(clean_3)


