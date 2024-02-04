def test_1():
    import spacy
    nlp=spacy.load('en_core_web_sm')
    doc = nlp(u'the horse galloped down the field and past the river.')
    sentence = []
    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article!
        if w.text != 'n' and not w.is_stop and not w.is_punct and not w.like_num:
            # we add the lematized version of the word
            sentence.append(w.lemma_)
    print(sentence)
    from gensim import corpora
    dictionary = corpora.Dictionary(texts)
    print(dictionary.token2id)
    #Let's say we would like to get rid of words that occur in less than 20 documents,
    # or in more than 50% of the documents, we would add the following:
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

from gensim import models
   tfidf = models.TfidfModel(corpus)
import gensim
   bigram = gensim.models.Phrases(texts)
texts = [bigram[line] for line in texts]


import nltk
   text = nltk.word_tokenize("And now for something completely different")
   nltk.pos_tag(text)
def training_pos():
    nlp = spacy.blank(lang)
    tagger = nlp.create_pipe('tagger')

    for tag, values in TAG_MAP.items():
        tagger.add_label(tag, values)
    nlp.add_pipe(tagger)

    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update([text], [annotations], sgd=optimizer, losses=losses)
        print(losses)


 from nltk.chunk import conlltags2tree, tree2conlltags
   from nltk import pos_tag
   from nltk import word_tokenize
   from nltk.chunk import ne_chunk

sentence = "Clement and Mathieu are working at Apple."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
iob_tagged = tree2conlltags(ne_tree)
print(iob_tagged)


for token in sent_0:
       print(token.text, token.ent_type_)



def dependancy_parsing():
    for token in sent_0:
        print(token.text, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children])
def lda():
    >> > from sklearn.decomposition import LatentDirichletAllocation
    >> > from sklearn.datasets import make_multilabel_classification
    >> >  # This produces a feature matrix of token counts, similar to what
    >> >  # CountVectorizer would produce on text.
    >> > X, _ = make_multilabel_classification(random_state=0)
    >> > lda = LatentDirichletAllocation(n_components=5,
                                         ...
    random_state = 0)
    >> > lda.fit(X)
    LatentDirichletAllocation(...)
    >> >  # get topics for some given samples:
    >> > lda.transform(X[-2:])
    array([[0.00360392, 0.25499205, 0.0036211, 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586]])



def test_kl():
    import numpy as np
    from scipy import stats

    # Generate sample data
    mean_1, std_1 = 5, 2
    samples1 = np.random.normal(loc=mean_1, scale=std_1, size=10000)

    mean_2, std_2 = 5, 2
    samples2 = np.random.normal(loc=mean_2, scale=std_2, size=10000)

    # Estimate distributions from samples
    dist1 = stats.norm.fit(samples1)
    dist2 = stats.norm.fit(samples2)

    # Compute KL divergence
    kl_divergence = stats.entropy(pk=dist1, qk=dist2)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    print(kl_divergence)

def tes_kl2():
    import numpy as np
    from scipy.stats import entropy
    from sklearn.neighbors import KernelDensity

    # Generate sample data
    mean_1, std_1 = 0, 1
    samples1 = np.random.normal(loc=mean_1, scale=std_1, size=10000)

    mean_2, std_2 = 0, 1
    samples2 = np.random.normal(loc=mean_2, scale=std_2, size=10000)

    # Fit kernel density estimators
    kde1 = KernelDensity(bandwidth=0.5).fit(samples1[:, np.newaxis])
    kde2 = KernelDensity(bandwidth=0.5).fit(samples2[:, np.newaxis])

    # Evaluate densities on a grid
    x = np.linspace(min(min(samples1), min(samples2)), max(max(samples1), max(samples2)), 1000)
    log_dens1 = kde1.score_samples(x[:, np.newaxis])
    log_dens2 = kde2.score_samples(x[:, np.newaxis])

    # Compute entropy
    entropy1 = entropy(np.exp(log_dens1))
    entropy2 = entropy(np.exp(log_dens2))

    # Calculate KL divergence
    kl_divergence = entropy1 - entropy2

    print("KL Divergence between two samples:", kl_divergence)

# ipython -i -m IdxSEC.bookNLP.test_1
if __name__=='__main__':
    print('e')