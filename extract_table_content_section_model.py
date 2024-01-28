import pandas as pd
import numpy as np
import os
from . import g_edgar_folder
from .model_util import save_sklearn_model,load_sklearn_model
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from .extract_table_content_section_supervised import g_table_content_section_training
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# this is the model for finding the ownership section in a table of section name


def train_model_v1():
    df=pd.read_csv(g_table_content_section_training)
    # TODO: add position in the document?

    df['section_num']=pd.to_numeric(df['section'],errors='coerce')
    df=df[pd.isna(df['section_num'])]
    df = df[~pd.isna(df['section'])]
    df['section']=df['section'].astype(str)

    train = df['section'].tolist()
    y = df['target'].tolist()

    #tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidfvectorizer = CountVectorizer(analyzer='word', stop_words='english')# better
    tfidf_wm = tfidfvectorizer.fit_transform(train)
    # this above is a sparse matrix
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    ntfidf_tokens=['w_'+x for x in tfidf_tokens]
    # index = ['Doc1','Doc2'],
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns =ntfidf_tokens ,index=df.index)
    print('Most frequent words : ')
    print(df_tfidfvect.mean().sort_values().tail(20))

    # training a model
    df = pd.concat([df,df_tfidfvect],axis=1,sort=False)

    X = df[ntfidf_tokens]
    y = df[['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model=DecisionTreeClassifier(max_depth=None)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_train)
    score=accuracy_score(y_pred, y_train)
    print('Classification score train: %.2f'%score)
    y_pred = model.predict(X_test)
    score=accuracy_score(y_pred, y_test)
    print('Classification score test: %.2f'%score)
    print(pd.Series(model.feature_importances_, index=X_train.columns).sort_values().tail())

    save_sklearn_model(tfidfvectorizer, 'table_content_section_tfidf')
    save_sklearn_model(model,'table_content_section')

def predict_table_content_section(df,debug=1):
    df0=df.copy()
    train = df['section'].tolist()
    tfidfvectorizer = load_sklearn_model('table_content_section_tfidf')
    tfidf_wm = tfidfvectorizer.transform(train)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    ntfidf_tokens=['w_'+x for x in tfidf_tokens]
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns =ntfidf_tokens ,index=df.index)

    df = pd.concat([df,df_tfidfvect],axis=1,sort=False)
    X = df[ntfidf_tokens]
    model=load_sklearn_model('table_content_section')
    res=model.predict(X)

    if debug>0:
        df0['t'] = res
        print(df0[['section','t']])
    return res

# ipython -i -m IdxSEC.extract_table_content_section_model
if __name__=='__main__':
    train_model_v1()





