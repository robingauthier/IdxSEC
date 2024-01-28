import hashlib
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('expand_frame_repr', False)


def myhash(txt):
    hash_obj = hashlib.sha256(txt.encode())
    hex_hash = hash_obj.hexdigest()
    return hex_hash[:10]

# where edgar raw data will be stored
g_edgar_folder = '/Users/sachadrevet/src/IdxSEC/data/'
g_model_folder = '/Users/sachadrevet/src/IdxSEC/model/'
g_ndata_folder = '/Users/sachadrevet/src/IdxSEC/data_generated/'


