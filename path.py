
import os
import getpass

# Windows+R
#  \\10.202.69.2\mktdata@STOXX@BASIC
# Credentials:
# User: abboud_ro
# Pw: 46N28a*7
mountdict ={
    'MSCI_DM_CORE_INDEXES':'P',
    'MSCI_DM_SMALL_CAP_INDEXES':'Q',
    'MSCI_EM_CORE_INDEXES':'S',
    'MSCI_EM_SMALL_CAP_INDEXES':'T',
    'FACTSET':'V',
    'NFACTSET':'R',
    'RUSSELL_FTSE_UK':'W',
    'BARRA_EUE4BAS':'E',
    'BARRA_GEMTR':'F',
    'BARRA_AUE4S':'W',
    'BARRA_ASE2':'M',
    'STOXX':'N',
    'UBS_SHORT':'G',
}
def get_datadir():
    uname=getpass.getuser()
    if os.name=='nt':
        return 'C:\\Users\\%s\\data\\'%uname
    elif os.name=='posix':
        return '/Users/sachadrevet/data/'
    else:
        raise(ValueError('Unknown OS type in get_datadir'))
def get_mountdir(name=''):
    if os.name=='nt':
        return mountdict[name]+':\\'
    elif os.name=='posix':
        return os.path.join(*['/mnt',name])
    else:
        raise(ValueError('Unknown OS type in get_mountdir'))
def get_codedir():
    if os.name=='nt':
        return 'C:\\Users\\sdrevet\\PycharmProjects\\'
    elif os.name=='posix':
        return '/home/sdrevet/src/'
    else:
        raise(ValueError('Unknown OS type in get_codedir'))

if __name__=='__main__':
    os.listdir(get_mountdir('MSCI_DM_CORE_INDEXES'))

