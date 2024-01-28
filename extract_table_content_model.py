import pandas as pd
import numpy as np
import os
from . import g_edgar_folder
from sklearn.metrics import accuracy_score
from .extract_table_content_supervised import g_table_content_training
from .model_util import load_sklearn_model,save_sklearn_model
def benchmark_model(resd):
    score = 0
    if resd['pos'] <= 10:
        score += 1
    if resd['pos'] > 40:
        score -= 1
    if resd['poslarge'] <= 5:
        score += 1
    if resd['poslarge'] > 5:
        score -= 1
    if resd['ncols'] >= 3:
        score -= 1
    if resd['has_text1']>0:
        score += 1
    if resd['nhref'] > 2:
        score += 1
    if resd['nhref'] > 10:
        score += 1
    if resd['nhref'] > 20:
        score += 1
    if resd['score_txt_p'] >= 5:
        score += 1
    if resd['score_txt_p'] >= 10:
        score += 1
    if resd['score_txt_n'] >= 5:
        score -= 1
    if resd['score_txt_n'] >= 10:
        score = 0
    if resd['nhref'] == 0:
        score = 0
    resd['score'] = score
    return score

def score_benchmark_model():
    df=pd.read_csv(g_table_content_training)
    df['score']=0
    for id,row in df.iterrows():
        df.loc[id,'score']=benchmark_model(row)
    df['pred']=np.where(df['score']>0,1,0)
    #print(df.sort_values(['pred','target']))
    score=accuracy_score(df['pred'], df['target'])
    print('Classification score : %.2f'%score)

def train_tree():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    df = pd.read_csv(g_table_content_training)
    X = df.drop(['fname','hash','target','isOwnership'],axis=1)
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
    print(pd.Series(model.feature_importances_,index=X_train.columns).sort_values())
    save_sklearn_model(model,'table_content')

def predict_table_content(df):
    model=load_sklearn_model('table_content')
    return model.predict(df.drop(['fname','hash'],axis=1))

    

# ipython -i -m IdxSEC.extract_table_content_model
if __name__=='__main__':
    #score_benchmark_model()
    train_tree()
