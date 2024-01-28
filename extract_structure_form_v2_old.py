import os.path
import numpy as np
import pandas as pd
import json
import re
from lxml import etree
from . import g_edgar_folder
from .extract_structure_form_v1 import extract_structure_form


def list_element_parents(element):
    root = element.getroottree()
    lr=[]
    tmp=element
    while tmp!=None:
        lr+=[tmp]
        tmp=tmp.getparent()
    return lr

def filter_html(fname, fromid=100, toid=1000,debug=1):
    """
    very powerful function that enables to filter an html document and only keep the middle of it
    """
    # I actually use the level information for speed reasons.
    structd = extract_structure_form(fname)

    with open(g_edgar_folder + fname, 'rt') as f:
        ff = f.read()
    restree = etree.HTML(ff)

    new_root = etree.Element("html")


    pelement=None
    elemid = 0
    #import pdb;pdb.set_trace()
    for element in restree.iter():
        elemid += 1

        hashloc=hash(etree.tostring(element, method='html', encoding=str))
        assert hashloc==structd[elemid]['hash'],'mismatch on the hash'

        if elemid < fromid or elemid > toid:
            continue

        if elemid==fromid:
            # we need to create the same structure as the current path
            if debug>0:
                print('element:      ' + restree.getroottree().getpath(element))
            list_parents=list_element_parents(element)[:-1]
            new_root_loc = new_root
            for par in reversed(list_parents):
                new_root_loc = etree.SubElement(new_root_loc, par.tag)
            if debug > 0:
                print('new_element:  ' + new_root.getroottree().getpath(new_root_loc))
                print('-'*10)

        if (debug>0) and (pelement is not None):
            print('pelement:  '+restree.getroottree().getpath(pelement))
            print('element:  '+restree.getroottree().getpath(element))
            print('level pelement : %i'%structd[elemid-1]['level'])
            print('level element :  %i'%structd[elemid]['level'])

        element_level=structd[elemid]['level']
        pelement_level = structd[elemid-1]['level'] if elemid>0 else None

        # Determining who is the new root
        if pelement is None:
            if debug>0:
                print('start condition')
            new_root_loc=new_root_loc.getparent()
        elif element_level == pelement_level:
            if debug > 0:
                print('same level')
            new_root_loc = new_element.getparent()
        elif pelement_level>element_level:
            nb_up=pelement_level-element_level
            if debug > 0:
                print('up %i level'%nb_up)
            new_root_loc = new_element
            for i in range(nb_up+1):
                new_root_loc=new_root_loc.getparent()
        elif pelement_level<element_level:
            nb_dn=-1*(pelement_level-element_level)
            if debug > 0:
                print('down %i level'%nb_dn)
            # we can only go down 1 level
            new_root_loc = new_element
        else:
            raise(ValueError('cannot happen'))

        try:
            new_element = etree.SubElement(new_root_loc, str(element.tag))  # ,attrib=element.attrib,nsmap=element.nsmap
        except Exception as e:
            continue
        new_element.text = element.text

        if debug>0:
            print('new_element :  '+new_root.getroottree().getpath(new_element))
            print('-'*10)


        if elemid>fromid+100:
            assert new_element.tag==element.tag,'issue'
            assert new_element.getparent().tag == element.getparent().tag, 'issue'

        pelement = element


    new_html = etree.tostring(new_root, method='html', encoding=str)
    return new_html



# ipython -i -m IdxSEC.extract_structure_form_v1
if __name__=='__main__':
    print('ok')
