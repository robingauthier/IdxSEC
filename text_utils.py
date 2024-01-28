import pandas as pd
import numpy as np
import re

# The character \xa0 represents a non-breaking space in Unicode.
# It is commonly referred to as a "no-break space" or "NBSP." In HTML,
# it is often represented as &nbsp;. This character is used to create
# a space between words that prevents line breaks at that position.

# The character \u200b represents a zero-width space in Unicode. It is also
# known as "zero-width space" or "ZWSP." As the name suggests, this character
# doesn't have a visible width and is typically used to indicate word boundaries
# without adding any visible space.
def preprocess_text(text):
    # took from ppytorch book
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a­zA­Z.,!?]+", r" ", text)
    return text
def clean_numbers(txtloc=' 6,943,654',meta=False):
    # to_numeric does not like commas
    txtloc_c=txtloc
    txtloc_c=txtloc_c.replace(',','')
    txtloc_c=txtloc_c.replace('%','')
    test=pd.to_numeric(txtloc_c,errors='coerce')
    if not pd.isna(test):
        if meta:
            return '[N]'
        else:
            return test
    if meta:
        return np.nan
    else:
        return txtloc

def clean_txt_v2(txtloc):
    #if 'of Certain Beneficial Owners and Management' in txtloc:
    #    import pdb;pdb.set_trace()
    txtloc = txtloc.replace('&nbsp;',' ')
    txtloc = txtloc.replace('\xa0', ' ')
    txtloc=txtloc.encode("ascii", "ignore").decode('utf-8')
    txtloc=txtloc.replace('\n',' ').replace('\t',' ')# faut remplacer par un espace
    txtloc = txtloc.replace('  ', ' ')
    txtloc = txtloc.replace('  ', ' ')
    return txtloc

def clean_txt_v0(section):
    section = re.sub(r'\s+', ' ', section)
    section = section.replace('\n', '')
    section = section.replace('\t', '')
    return section
def is_meta(txt):
    cond1=str(txt)[0] == '['
    cond2 = str(txt)[-1] == ']'
    return cond1 and cond2

def clean_txt(txt,meta=False):
    if txt is None:
        return ''
    if not isinstance(txt,str):
        return txt
    txt = txt.replace('\xa0',' ') # this is a non breaking space
    txt = txt.replace('<br>', ' ')  # new line in html
    txt = re.sub(r'[^\x00-\x7F]+', '', txt)  # removes all unicode characters
    txt = re.sub(r'\s+', ' ', txt) # replaces multiple space by 1 space

    if txt==' ' or txt=='':
        if meta:
            return '[E]'
        else:
            return ''
    txt = clean_numbers(txt,meta=meta)
    if txt==' ' or txt=='':
        if meta:
            return '[E]'
        else:
            return ''
    if meta:
        if is_meta(txt):
            return txt
        else:
            return '[T]'
    else:
        return txt

def remove_duplicated_tags(txt):
    if '[B]' in str(txt):
        return str(txt).replace('[B]','')+'[B]'
    return txt

def test_1():
    res=clean_numbers(txtloc=' 6,943,654.1')
    assert res==6943654.1,'issue'
    res=clean_numbers(txtloc=' 6,943,654.1',meta=True)
    assert res=='[N]','issue'
    res=clean_txt(txt=' 6,943,654.1',meta=True)
    assert res=='[N]','issue'
    res = clean_numbers(txtloc='h 6,943,654.1')
    assert res=='h 6,943,654.1','issue'
    res= clean_txt(txt='\xa0')
    assert res=='','issue'
    res = clean_txt(txt='\u200b')
    assert res == '', 'issue'
    res = clean_txt(txt='\xa0\xa0')
    assert res == '', 'issue'
    res = clean_txt(txt='<br>')
    assert res == '', 'issue'
    res = clean_txt(txt='Shares of<br>Class A')
    assert res == 'Shares of Class A', 'issue'
    res = clean_txt(txt='Shares\xa0of\xa0Common\xa0Stock\xa0Beneficial\xa0Ownership [B]')
    assert res == 'Shares of Common Stock Beneficial Ownership [B]', 'issue'
    res = clean_txt(txt='Shares\xa0of\xa0Common\xa0Stock\xa0Beneficial\xa0Ownership [B]',meta=True)
    assert res == '[T]', 'issue'
    res = clean_txt(txt='\u200b',meta=True)
    assert res == '[E]', 'issue'
    res=clean_txt(txt=' ',meta=True)
    assert res=='[E]','issue'

def re_search_all(regex, text, gname):
    matches = re.finditer(regex, text)
    lr = []
    for match in matches:
        lr += [{'match': match.group(gname),
                'startpos': match.start(),
                'endpos': match.end()}]
    return pd.DataFrame(lr)

    #import pdb;pdb.set_trace()
# ipython -i -m IdxSEC.text_utils
if __name__=='__main__':
    test_1()
    #res = clean_txt(txt='\u200b', meta=True)