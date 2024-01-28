


def score_text_list(txtloc,lpos,uselower=True):
    if uselower:
        txtlocl = txtloc.lower()
    else:
        txtlocl = txtloc
    scorep=0
    for wpos in lpos:
        scorep+=txtlocl.count(wpos)
    return scorep
