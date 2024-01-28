import pandas as pd
import numpy as np
import os
from . import g_edgar_folder
from .scoring_utils import score_text_list
lpos=['security ownership of management and certain beneficial owners',
      'security ownership',
      'beneficial ownership',
      'five percent',
      'ownership of securities',
      '5 percent',
      'ownership']
lneg=[
    'compliance','matters','guidelines','alignment','retention','policy'
]
def predict_table_content_section(df,debug=0):
    """returns a serie with 0/1 1 for section is ownership and 0 if other

    """
    df0=df.copy()
    for id,row in df0.iterrows():
        scorep=score_text_list(row['section'],lpos)
        scoren = score_text_list(row['section'], lneg)
        if scoren>0:
            score=0
        else:
            score=scorep
        df0.loc[id,'score']=score
    if debug>0:
        print('predict_table_content_section::debug::')
        print(df0[['section','score']])
    #import pdb;pdb.set_trace()
    df0['score']=np.sign(df0['score'])
    return df0['score']

# ipython -i -m IdxSEC.extract_table_content_section_model
if __name__=='__main__':
    print('ready')





