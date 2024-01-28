from numba import jit
lpos = [
    'beneficial owners',
    'beneficial ownership',
    'director and executive officer ownership',
    'executive officer and director stock ownership',
    'five percent holders',
    'greater than 5% security holders',
    'holders of stock',
    'management ownership',
    'ownership by directors',
    'ownership of certain beneficial owners',
    'ownership of equity securities',
    'ownership of principal stockholders',
    'ownership of voting securities',
    'principal holders of stock',
    'principal shareholders',
    'principal stockholders',
    'security ownership',
    'significant shareholders',
    'stock owned by directors',
    'stock ownership'
        ]
lneg = [
'additional information',
 'additional voting matters',
 'agenda item',
 'annual meeting information',
 'audit committee report',
 'audit matters',
 'awards granted',
 'board and committee matters',
 'ceo pay ratio',
 'code of ethics',
 'compensation discussion & analysis',
 'compensation discussion and analysis',
 'compensation tables',
 'corporate governance guidelinespension benefits',
 'director compensation',
 'director stock ownership requirement',
 'equity compensation',
 'equity incentive',
 'executive compensation table',
 'option exercises and stock vested',
 'outstanding equity awards',
 'plan benefits',
 'proxy summary',
 'related-party transactions',
 'relationships and related party transactions',
 'report of the hrc committee',
 'role of the compensation committee',
 'shareholder proposals ',
 'stock ownership guidelines',
 'stock ownership policy',
 'stockholder approval',
 'summary compensation table'
]

def score_text_list(txtloc,lpos,lneg):
    txtlocl = txtloc.lower()
    scorep=0
    scoren = 0
    for wpos in lpos:
        scorep+=txtlocl.count(wpos)
    for wneg in lneg:
        scoren+=txtlocl.count(wneg)
    return {'scorep':scorep,'scoren':scoren}

# ipython -i -m IdxSEC.score_text_ownership
if __name__=='__main__':
    txt=('The following table sets forth certain information regarding the beneficial ownership '
         'of our common stock as of February 22, 2023 (except as otherwise indicated) by (i) each'
         ' person or entity known by us to beneficially own more than 5% of our common stock, '
         '(ii) each director, (iii) each ex')
    resd=score_text_ownership(txt)
