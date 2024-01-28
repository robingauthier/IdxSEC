import pandas as pd
from .edgar_utils import convert_to_sec_url
from .extract_structure_form import extract_structure_form
from . import g_edgar_folder

def format_nicely(fname):
    url = convert_to_sec_url(fname)
    print('Fname is:')
    print(fname)
    print('Pls check for the online version : ')
    print(url)
    sdf=extract_structure_form(fname)
    for rcol in ['fname','fileid']:
        if rcol in sdf.columns:
            sdf=sdf.drop([rcol],axis=1)
    #parentids=sdf[sdf['score1']>0]['parent1'].tolist()
    #sdf['sel']=-1*sdf['parent1'].isin(parentids)
    sdf['sel']=0
    sdf=sdf.sort_values(['sel','elemi'],ascending=True)
    print('Look at this file :')
    print(g_edgar_folder+'../data_supervised/structure_test.txt')
    with open(g_edgar_folder+'../data_supervised/structure_test.txt', 'w') as outfile:
        sdf.to_string(outfile)

# ipython -i -m IdxSEC.manually_tag_form_structure
if __name__=='__main__':
    from .filesys import find_files_using_glob
    print(g_edgar_folder+'form14A_cik*.txt')
    lfilesdf=find_files_using_glob(g_edgar_folder+'form14A_cik*.txt')
    lfiles=list(set(lfilesdf['file'].tolist()))

    #fname='form14A_cik14693_asof20230623_0001193125-23-173793.txt'
    fname=lfiles[250]
    #fname='form14A_cik1907982_asof20230424_0001140361-23-019759.txt'
    #fname='form14A_cik1042729_asof20230414_0001437749-23-010228.txt'
    #fname = 'formA14A_cik1018724_asof20230503_0001104659-23-055145.txt'
    #fname = 'form14A_cik898174_asof20230413_0000898174-23-000081.txt'
    #fname = 'form14A_cik1819576_asof20230428_0001104659-23-053124.txt'
    #fname='form14A_cik1864032_asof20230601_0001214659-23-008175.txt'# ok
    #fname='form14A_cik1166388_asof20230511_0001166388-23-000080.txt'
    #fname='form14A_cik838875_asof20230531_0001199835-23-000300.txt'
    format_nicely(fname)
