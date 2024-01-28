from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from . import g_edgar_folder

# https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
# tf-idf is a measure of originality of a word by comparing
# the number of times the word appears in a doc versus
# the number of doc the word appears in.
# TfidfVectorizer,CountVectorizer : CountVectorizer does not normalize by doc frequency


def test_sbert():
    # https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    # Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")

# /Users/sachadrevet/src/IdxSEC/extract_table_content_model_tfidf.py
# ipython -i -m IdxSEC.extract_table_content_model_tfidf
if __name__=='__main__':
    fname = g_edgar_folder + '../data_supervised/table_content_all.txt'
    df=pd.read_csv(fname)
    df['section_num']=pd.to_numeric(df['section'],errors='coerce')
    df=df[pd.isna(df['section_num'])]
    df = df[~pd.isna(df['section'])]
    #df = df.drop_duplicates(subset=['href','fname','hash'])
    df['section']=df['section'].astype(str)
    debug=0
    train=[]
    for groupid,dfloc in df.groupby(['fname','hash']):
        txtloc = ' '.join(dfloc['section'].tolist())
        txtloc+= ' '+dfloc['href'].iloc[0]
        if debug>0:
            print(groupid)
            print(dfloc.head())
            print(txtloc)
            print('-' * 20)
        train+=[txtloc]
    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    #tfidfvectorizer = CountVectorizer(analyzer='word', stop_words='english')
    tfidf_wm = tfidfvectorizer.fit_transform(train)
    # this above is a sparse matrix
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    # index = ['Doc1','Doc2'],
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
    print('Most frequent words : ')
    print(df_tfidfvect.mean().sort_values().tail(20))


    # equity          0.396811
    # 2022            0.410650
    # report          0.411853
    # certain         0.412304
    # plan            0.416968
    # audit           0.418321
    # governance      0.450211
    # vote            0.469013
    # meeting         0.520307
    # annual          0.528430
    # information     0.564531
    # director        0.645156
    # committee       0.662906
    # contents        0.680054
    # table           0.849128
    # directors       0.853039
    # executive       1.041667
    # board           1.081829
    # proposal        1.099278
    # compensation    2.113568

    # new_id          0.024809
    # 2022            0.025085
    # certain         0.025518
    # meeting         0.027915
    # annual          0.028817
    # information     0.031479
    # committee       0.032474
    # director        0.033220
    # directors       0.041688
    # executive       0.050122
    # references      0.051384
    # board           0.051760
    # proposal        0.056153
    # javascript      0.072677
    # void            0.072677
    # details         0.073299
    # toc             0.085424
    # compensation    0.089280
    # table           0.099252
    # contents        0.112162