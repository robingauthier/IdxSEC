import pandas as pd
import numpy as np
import os
from . import g_edgar_folder
from .filesys import append_to_file
from .extract_table_content_v1 import extract_table_content
from .extract_table_content_supervised import mydatadf
import distance

g_table_content_section_training = g_edgar_folder + '../data_supervised/table_content_section_training.txt'

def create_target_data(reset=True):
    """
    Target data for classifying if this is the ownership section or not
    we also feed tables that are not content tables
    we rely on the field isOwnership
    """
    if reset:
        try:
            os.remove(g_table_content_section_training)
        except Exception as e:
            pass

    lfiles = list(set(mydatadf['fname'].tolist()))
    for fname in lfiles:
        mydatadfloc=mydatadf.loc[lambda x:x['fname']==fname].copy()
        mydatadfloc['target']=1.0
        mydatadfloc=mydatadfloc.rename(columns={'section':'section1'})

        searchlogdf, content_table = extract_table_content(fname)
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

        ncontent_table['isOwnership']=ncontent_table['isOwnership'].fillna(0.0)
        ncontent_table['target']=ncontent_table['target'].fillna(0.0)
        ncontent_table['target'] = np.where(ncontent_table['dst']>0.2,0,ncontent_table['target'])
        ncontent_table['target'] = ncontent_table['isOwnership']*ncontent_table['target']

        if False:
            ncontent_table['section']=ncontent_table['section'].str[:20]
            ncontent_table['section1'] = ncontent_table['section1'].str[:20]
            print(ncontent_table[['section','section1','target','dst','isOwnership']])
        ncontent_table=ncontent_table.drop(['dst','isOwnership'],axis=1)

        append_to_file(ncontent_table, g_table_content_section_training)


# ipython -i -m IdxSEC.extract_table_content_section_supervised
if __name__=='__main__':
    #print(mydatadf)
    #test_1()
    #test_2()
    if True:
        create_target_data(reset=True)
    if True:
        df=pd.read_csv(g_table_content_section_training)
        print(df.shape[0])
        print(df[df['target']>0][['section','href']])