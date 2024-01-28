import gensim
import pandas as pd

from .corpus_create_v1 import batch_iterator
from .corpus_create_v1 import get_tokenizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
from . import g_ndata_folder
from . import g_model_folder
import random

random.seed(42)

# https://radimrehurek.com/gensim/models/ldamodel.html

def old():
    lda = LdaModel(common_corpus, num_topics=10)
    no_topic = 10
    nmf = NMF(n_components=no_topic).fit(tfidf_corpus)
    lda = LatentDirichletAllocation(n_topics=no_topics).fit(tf_corpus)
    pyLDAvis.gensim.prepare(model, corpus, dictionary)
def test_1():
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    # Sample documents
    documents = [
        "apple banana mango",
        "orange banana apple",
        "banana mango banana"
    ]

    # Tokenize documents
    tokenized_documents = [doc.split() for doc in documents]

    # Create dictionary and corpus
    dictionary = Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
    # [[(0, 1), (1, 1), (2, 1)], [(0, 1), (1, 1), (3, 1)], [(1, 2), (2, 1)]]

    # Train LDA model
    num_topics = 2
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # Get topics
    topics = lda_model.print_topics(num_topics=num_topics)
    for topic in topics:
        print(topic)
    import pdb;pdb.set_trace()
def mylda():
    # Latent Dirichlet Allocation (LDA) is typically run on the bag-of-words
    # representation of the text data.
    tokenizer = get_tokenizer()

    dict_id2tok={v:k for k,v in tokenizer.get_vocab().items()}
    dict_tok2id = tokenizer.get_vocab()

    for ltxt in batch_iterator():
        lr=[]
        r=tokenizer.encode_batch(ltxt)
        # returns of list of Encode objects
        for rloc in r:
            rowd = {}
            tokloc=rloc.tokens
            nbtokens=len(tokloc)
            for tok in tokloc:
                if tok in rowd.keys():
                    rowd[tok]+=1/nbtokens
                else:
                    rowd[tok]=1/nbtokens
            lr+=[rowd]
        rdf=pd.DataFrame(lr).fillna(0.0)
        ldamodel,e2 = gensim.models.LdaModel(corpus=rdf, num_topics=10, id2word=dict_id2tok)
        import pdb;pdb.set_trace()
    #corpusloc = corpus[0]
    #
    # Update the model by incrementally training on the new corpus
    #lda.update(other_corpus)
    #ldamodel.show_topics()
    #model.get_term_topics('water')
def mylda2():
    # training takes 0 days 01:01:05.040840
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    import pyLDAvis.gensim_models
    import os
    os.environ['TOKENIZERS_PARALLELISM']='false'
    tokenizer = get_tokenizer()

    lda_model = None
    num_topics = 20

    dict_id2tok={v:k for k,v in tokenizer.get_vocab().items()}
    dict_tok2id = tokenizer.get_vocab()

    dictionary = Dictionary(documents=None)
    dictionary.token2id = dict_tok2id
    dictionary.id2token = dict_id2tok
    i=0
    st=pd.to_datetime('now')
    for formdoc in batch_iterator():
        i+=1
        if i%10==0:
            print('i=%i'%i)
        #if i>100:
        #    break
        tformdoc = tokenizer.encode_batch(formdoc)
        tokenized_formdoc=[x.tokens for x in tformdoc]
        corpus = [dictionary.doc2bow(tokenized_formpar) for tokenized_formpar in tokenized_formdoc]
        if lda_model is None:
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        else:
            lda_model.update(corpus=corpus)
    et = pd.to_datetime('now')
    print(et-st)
    # Get topics
    topics = lda_model.print_topics(num_topics=num_topics)
    print('-'*20)
    for topic in topics:
        print(topic)
    vis=pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, g_ndata_folder+'lda.html')
    print('Please open '+g_ndata_folder+'lda.html')

    lda_model.save(g_model_folder+'lda_form14.pkl')
    nmodel= LdaModel.load(g_model_folder+'lda_form14.pkl', mmap='r')
    import pdb;pdb.set_trace()

def test_3():
    """the html generated is easy to understand really"""
    # https://spacy.io/usage/visualizers
    from spacy import displacy
    ex = [{"text": "But Google is starting from behind.",
           "ents": [{"start": 4, "end": 10, "label": "ORG"}],
           "title": None},
          {"text": "Hello the world Google is starting from behind.",
           "ents": [{"start": 4, "end": 10, "label": "ORG2"}],
           "title": None},
          {"text": "Hello the world Google is starting from behind.",
           "ents": [{"start": 4, "end": 10, "label": "myLabel2"}],
           "title": None},
          ]
    colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    colors = {"ORG": "rgb(0, 255, 0)","ORG2": "rgb(0, 125, 0)"}
    options = {"ents": ["ORG","ORG2"], "colors": colors}
    #displacy.serve(doc, style="ent", options=options)
    html = displacy.render(ex, style="ent", manual=True,options=options)
    nfname=g_ndata_folder + 'spacy_annotation.html'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html)
    import pdb
    pdb.set_trace()
def topic_classifier(fname):
    from .blocks_html_v1 import blocks_html
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from gensim.parsing.preprocessing import preprocess_string
    import pyLDAvis.gensim_models
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = get_tokenizer()

    lda_model = LdaModel.load(g_model_folder+'lda_form14.pkl', mmap='r')

    dict_id2tok = {v: k for k, v in tokenizer.get_vocab().items()}
    dict_tok2id = tokenizer.get_vocab()

    dictionary = Dictionary(documents=None)
    dictionary.token2id = dict_tok2id
    dictionary.id2token = dict_id2tok

    lstruct, _ = blocks_html(fname)

    lr=[]
    for elem in lstruct:
        if elem['tag'] != 'table':
            tmp_tokens=preprocess_string(elem['content'])
            tmp_tokens = [x for x in tmp_tokens if len(x)<=15]
            lr += [' '.join(tmp_tokens)]

    formdoc=lr
    tformdoc = tokenizer.encode_batch(formdoc)
    tokenized_formdoc=[x.tokens for x in tformdoc]
    corpus = [dictionary.doc2bow(tokenized_formpar) for tokenized_formpar in tokenized_formdoc]

    r=lda_model.get_document_topics(corpus)
    # (Pdb) p r[19]
    # [(6, 0.13125263), (7, 0.12696739), (13, 0.1355281), (15, 0.1312486)

    nr=[]
    for rloc in r:
        rlocd = {k:v for k,v in rloc}
        nr+=[rlocd]
    rdf=pd.DataFrame(nr)
    maintopic = rdf.fillna(0).idxmax(axis=1)

    ltopics = lda_model.print_topics(num_topics=50)


    return {'topics':maintopic,'lstruct':lstruct,'ltext':formdoc,'model':lda_model,'ltopics':ltopics}

# Function to generate random RGB color
def random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgb({r}, {g}, {b})"

def visualize_topic_classifier(fname,nice=True):
    from spacy import displacy
    rd=topic_classifier(fname)
    topics=rd['topics'].unique().tolist()
    colors={}
    ents=[]
    for topiloc in topics:
        colors['T%i'%topiloc]=random_rgb()
        ents+=['T%i'%topiloc]


    if nice:
        ex=[]
        i=-1
        for elem in rd['lstruct']:
            if elem['tag'] == 'table':
                continue
            i+=1
            topicloc=rd['topics'].iloc[i]
            exloc={'text':'   '+elem['content'],
                   'ents':[{"start": 0, "end": 3, "label": "T%i"%topicloc}],
                   'title':None}
            ex+=[exloc]
    else:
        ex=[]
        i=-1
        for txt in rd['ltext']:
            i+=1
            topicloc=rd['topics'].iloc[i]
            exloc={'text':'   '+txt,
                   'ents':[{"start": 0, "end": 3, "label": "T%i"%topicloc}],
                   'title':None}
            ex+=[exloc]

    options = {"ents": ents, "colors": colors}
    html = displacy.render(ex, style="ent", manual=True,options=options)
    nfname=g_ndata_folder + 'spacy_annotation.html'
    print('Open text file '+nfname)
    with open(nfname, 'wt') as f:
        f.write(html)
    #import pdb;pdb.set_trace()


# ipython -i -m IdxSEC.topics_lda
if __name__=='__main__':
    #mylda2()
    #test_3()
    from .edgar_utils import get_random_form
    fname=get_random_form()
    #topic_classifier(fname)
    visualize_topic_classifier(fname)