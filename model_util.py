import joblib
import pickle
from . import g_model_folder

def save0_sklearn_model(model,name):
    fname=g_model_folder+'model_'+name+'.joblib'
    joblib.dump(model,fname)

def load0_sklearn_model(name):
    fname = g_model_folder + 'model_' + name + '.joblib'
    return joblib.load(fname)


def save_sklearn_model(model, name):
    fname = g_model_folder + 'model_' + name + '.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(model, file)

def load_sklearn_model(name):
    fname = g_model_folder + 'model_' + name + '.pkl'
    with open(fname, 'rb') as file:
        return pickle.load(file)