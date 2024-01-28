
import os
import shutil
from tqdm import tqdm
import pandas as pd
from glob import glob
import re
from datetime import datetime
from .path import get_datadir
from dateutil.relativedelta import relativedelta
# We created a way to save data day by day
g_data = get_datadir()
def print_orange(x):
    W = '\033[0m'  # white (normal)
    R = '\033[31m'  # red
    G = '\033[32m'  # green
    O = '\033[33m'  # orange
    B = '\033[34m'  # blue
    P = '\033[35m'  # purple
    print(O+x+W)

def get_fullname(filename):
    return os.path.join(g_data,filename)
def save_file(df,fnamepattern='blocktrds_%i.csv'):
    assert '%i' in fnamepattern,'issue in the name of files'
    i= 0
    done=False
    while (i <=1000) and (not done):
        fname = fnamepattern % i
        try:
            df.to_csv(fname)
            done=True
        except Exception as e:
            done=False
        i+=1
def UnZipFile(fname,checkfirst=False,tmpdir=None):
    """given a file which is a zip file we will copy it to a temp directory
    unzip it and return the name of the unzipped directory"""
    basefname = os.path.basename(fname)
    if tmpdir is None:
        tmpdir = get_fullname('tmp')
    target = os.path.join(tmpdir, basefname)
    targetunzip = target.replace('.zip', '').replace('.ZIP', '')
    # Checking if the result exists
    if checkfirst:
        targetexists = os.path.exists(targetunzip)
        if targetexists:
            print('UnZipFile::target already exists')
            return targetunzip
    # If tmpdir does not exists we create it
    try:
        os.mkdir(tmpdir)
    except Exception as e:
        pass
    if checkfirst:
        if not os.path.exists(target):
            shutil.copy(fname, target)
    else:
        shutil.copy(fname, target)
    os.chmod(target,0o775)
    print('Unzipping %s'%target)
    shutil.unpack_archive(target, extract_dir=targetunzip, format='zip')
    return targetunzip
def convert_date(date):
    if isinstance(date,pd.Timestamp):
        return date
    if isinstance(date,datetime):
        return date
    if isinstance(date, str):
        return pd.to_datetime(date)
def check_patternpath(patternpath):
    assert 'YYYY' in patternpath
    assert 'MM' in patternpath
    assert 'DD' in patternpath
def convert_patternpath_regexp(patternpath):
    return patternpath \
        .replace('\\','\\\\')\
        .replace('YYYYMMDD', '(?P<yyyymmdd>\d{8})')\
        .replace('YYYY','(?P<year>\d{4})')\
        .replace('MM','(?P<month>\d{2})')\
        .replace('DD','(?P<day>\d{2})')
def read_generic(file,compression=None):
    extension = file[-3:]
    if extension == 'csv':
        try:
            return pd.read_csv(file)
        except Exception as e:
            try:
                return pd.read_csv(file,encoding = 'ISO-8859-1')
            except Exception as e:
                return pd.DataFrame()
    elif extension =='pkl':
        if compression is None:
            return pd.read_pickle(file)
        else:
            try:
                return pd.read_pickle(file,compression=compression)
            except Exception as e:
                return pd.read_pickle(file)
    else:
        raise(ValueError('Unknown file type in read_generic'))
def subs_patternpath_date(patternpath,date):
    date=convert_date(date)
    return patternpath \
        .replace('YYYYMMDD', date.strftime('%Y%m%d'))\
        .replace('YYYY',date.strftime('%Y'))\
        .replace('MM',date.strftime('%m'))\
        .replace('DD',date.strftime('%d'))
def find_files(patternpath):
    founds  = []
    patternpathre = convert_patternpath_regexp(patternpath)
    dir = os.path.dirname(patternpath)
    for fname_ in os.listdir(dir):
        fname = os.path.join(dir,fname_)
        matchobj = re.match(patternpathre,fname)
        if matchobj is None:
            continue
        mtime=os.path.getmtime(fname)
        try:
            fdate = pd.to_datetime(matchobj.group('yyyymmdd'))
        except Exception as e:
            fdate = pd.to_datetime(matchobj.group('year')+'-'+matchobj.group('month')+'-'+matchobj.group('day'))
        founds+=[{'date':fdate,'file':fname,'mtime':mtime}]
    founds_df = pd.DataFrame(founds)
    return  founds_df

def find_files_using_glob(patternpath):
    founds  = []
    dir = os.path.dirname(patternpath)
    lfiles = glob(patternpath)
    if len(lfiles)==0:
        raise(ValueError('find_files_using_glob empty for '+patternpath))
    for fname_ in lfiles:
        fname = os.path.basename(fname_)
        mtime=os.path.getmtime(fname_)
        ctime = os.path.getctime(fname_)
        size = os.path.getsize(fname_)
        founds+=[{'file':fname,'mtime':mtime,'ctime':ctime,'size':size/1024,'dir':dir}]
    founds_df = pd.DataFrame(founds)
    founds_df['mtime'] = pd.to_datetime(founds_df['mtime']*1e9)
    founds_df['ctime'] = pd.to_datetime(founds_df['ctime']*1e9)
    return  founds_df

def find_last_file(patternpath,date=pd.to_datetime('now',utc=True),debug=False,adddate=False,lasti=1):
    """assuming we need the last available file before that date"""
    date=convert_date(date)
    founds_df = find_files(patternpath)
    if debug:
        print(founds_df)
    if founds_df.shape[0] == 0:
        if adddate:
            return None,None
        return None
    founds_df=founds_df[founds_df['date']<=date]
    if founds_df.shape[0] == 0:
        print_orange('Caution::find_last_file : no file found before %s '%date)
        return find_last_file(patternpath,date+relativedelta(months=3),debug=debug,adddate=adddate)
        #if adddate:
        #    return None,None
        #return None
    founds_df.sort_values(['date','mtime'],inplace=True)
    if adddate:
        return founds_df.iloc[-lasti]['file'],founds_df.iloc[-1]['date']
    return founds_df.iloc[-lasti]['file']
def find_all_path(path):
    listfiles=os.listdir(path)
    resl=[]
    for ff in listfiles:
        fullff=os.path.join(path,ff)
        resl+=[{'fullfile':fullff,'mtime':os.path.getmtime(fullff),'file':ff}]
    res=pd.DataFrame(resl).sort_values('mtime')
    res['mtime'] = pd.to_datetime(res['mtime'] * 1e9)
    return res
def find_last_path(path):
    listfiles=os.listdir(path)
    resl=[]
    for ff in listfiles:
        fullff=os.path.join(path,ff)
        resl+=[{'fullfile':fullff,'mtime':os.path.getmtime(fullff),'file':ff}]
    res=pd.DataFrame(resl).sort_values('mtime')
    lastfile=res['fullfile'].iloc[-1]
    print('Using file :%s'%lastfile)
    return lastfile
def find_next_file(patternpath,date):
    """assuming we need the last available file before that date"""
    date=convert_date(date)
    founds_df = find_files(patternpath)
    if founds_df.shape[0] == 0:
        return None
    founds_df=founds_df[founds_df['date']>=date]
    if founds_df.shape[0] == 0:
        return None
    founds_df.sort_values('date',inplace=True)
    return founds_df.iloc[-1]['file']
def find_last(patternpath,date):
    file= find_last_file(patternpath,date)
    return read_generic(file)
def read_data(patternpath,dts,filter=None,verbose=0,compression=None):
    """ Loop on all possible file names and concatenate"""
    fullpatternpath=get_fullname(patternpath)
    #sDate = convert_date(sDate)
    #eDate = convert_date(eDate)
    #if dts is None:
    #    dts = pd.bdate_range(sDate,eDate)
    dfl = []
    desc='Reading '+patternpath.replace('YYYYMMDD','').replace('.pkl','')
    for date in tqdm(dts,desc=desc):
        fname = subs_patternpath_date(fullpatternpath,date)
        if not os.path.exists(fname):
            if verbose>0:
                print('Missing file for %s'%date.strftime('%Y%m%d'))
            continue
        if filter is None:
            dfloc = read_generic(fname,compression)
        else:
            dfloc = read_generic(fname,compression).loc[lambda x:filter(x)]
        if 'date' not in dfloc.columns:
            dfloc['date']=date
        dfl += [dfloc]
    return pd.concat(dfl,axis=0,sort=False)
def read_datag(patternpath,filter=None,verbose=0,compression=None,maxfiles=None):
    """ Loop on all possible file names and concatenate"""
    dir = os.path.dirname(patternpath)
    try:
        filedf=find_files_using_glob(patternpath)
    except Exception as e:
        return pd.DataFrame()
    dfl = []
    desc='Reading '+patternpath
    i = 0
    for id,row in tqdm(filedf.iterrows(),desc=desc):
        i+=1
        if maxfiles is not None:
            if i>maxfiles:
                print_orange('Caution read_datag stopping the read at maxfiles = %i'%maxfiles)
                break
        fname = os.path.join(dir,row['file'])
        if not os.path.exists(fname):
            if verbose>0:
                print('Missing file for %s'%date.strftime('%Y%m%d'))
            continue
        if filter is None:
            dfloc = read_generic(fname,compression)
        else:
            dfloc = read_generic(fname,compression).loc[lambda x:filter(x)]
        if dfloc.shape[0]==0:
            continue
        dfloc['fname']=fname
        #print(fname)
        #print(dfloc.head().iloc[:,:10])
        dfl += [dfloc]
    return pd.concat(dfl,axis=0,sort=False)


def append_to_file(df, fname):
    """ we will append to a file here"""
    if not os.path.exists(fname):
        df.to_csv(fname, index=False)
        return
    refdf=pd.read_csv(fname,nrows=3)
    refcols0 = refdf.columns.tolist()
    refcols = sorted(refdf.columns.tolist())
    for col in refcols:
        if col not in df.columns:
            df[col]=np.nan
    cols = sorted(df.columns.tolist())
    if refcols!=cols:
        print('Caution we have an issue on the columns on %s -- Erasing & recreating file'%fname)
        shutil.copy(fname,fname.replace('.csv','old%s.csv'%pd.to_datetime('now').strftime('%Y%m%d%H%M%S')))
        #dfi = pd.read_csv(fname)
        #dfr = pd.concat([dfi,refdf],axis=0,sort=False)
        df.to_csv(fname, index=False)
    else:
        df[refcols0].to_csv(fname, index=False,mode='a',header=False)

# ipython -i -m CommonLib.filesys
if __name__=='__main__':
    #from .path import get_datadir
    #from glob import glob
    #df=find_files_using_glob(get_fullname('mhprofiler_*.pkl'))
    #df=read_datag('D:\\data\\jpmorgan_*_focus_list.csv')
    df=find_files_using_glob('D:\\data\\tickkdb_v5\\barmulti_stock_v5_min=2_20220[(01)|(02)|(11)]*.csv')
    #for file in glob('U:\\TeamAbboud_Shared\\SP_Capital_IQ\\*.zip'):
    #    shutil.unpack_archive(file, extract_dir=file.replace('.zip',''), format='zip')
    #g_data = get_datadir()
    #patternpath = os.path.join(g_data,'lse_compositioninstrumentlist_YYYYMMDD.csv')
    #r = find_last_file(patternpath,'2018-05-01')
    #shutil.unpack_archive(target, extract_dir=targetunzip, format='zip')
    #df=read_data(patternpath,'2018-05-01','2019-12-01')

