
from .convert_html_text import convert_html_to_text

# alphabet form we need to make sure the ownership table looks good
fname='form14A_cik1652044_asof20230421_0001308179-23-000736.txt'

#def test():

if __name__=='__main__':
    fname = 'form14A_cik1652044_asof20230421_0001308179-23-000736.txt'
    txt=convert_html_to_text(fname)

