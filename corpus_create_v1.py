from .edgar_utils import get_forms
from .blocks_html_v1 import blocks_html
from . import g_edgar_folder
from . import g_model_folder
from gensim.parsing.preprocessing import preprocess_string

# https://github.com/huggingface/tokenizers/tree/main
# rust code : https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/trainer.rs
# https://huggingface.co/docs/tokenizers/pipeline
# https://huggingface.co/docs/tokenizers/training_from_memory

def get_form_txt(fname):
    # https://radimrehurek.com/gensim/parsing/preprocessing.html
    lr = []
    lstruct, _ = blocks_html(fname)
    for elem in lstruct:
        if elem['tag'] != 'table':
            #lr += [elem['content']]
            tmp_tokens=preprocess_string(elem['content'])
            # we need to remove super long tokens like
            # applicablefydeterminedbasedonchangeinascfairvaluefrompriorfyend
            tmp_tokens = [x for x in tmp_tokens if len(x)<=15]
            lr += [' '.join(tmp_tokens)]
            #import pdb;pdb.set_trace()
    return lr

def batch_iterator():
    lfiles = get_forms()#[:100]
    print('nb of files :%i'%len(lfiles))
    for fname in lfiles:
        #print(fname)
        yield get_form_txt(fname)

def train_tokenizer_bpe():
    # 45 minutes to train for 1600 forms
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    )
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(g_model_folder+'tokenizer_bpe_form14.json')
    #new_tokenizer = Tokenizer.from_file("tokenizer.json")
    #p tokenizer.get_vocab()
    #e=tokenizer.encode('First they generate a contextual embedding for each input token by an autoencoder.')
    #e = tokenizer.encode("Let's test this tokenizer.")
    #print(e.tokens)
    #p tokenizer.decode(e.ids)
    import pdb;pdb.set_trace()

def get_vocab_frequency():
    # Ġexecut       170145
    # Ġproxi        173067
    # Ġannual       184319
    # Ġstock        210307
    # Ġcommitte     211309
    # Ġmeet         213383
    # Ġcompens      215274
    # Ġvote         235432
    # Ġboard        247261
    # Ġdirector     274409
    # Ġshare        278254
    # Ġcompani      294672
    import pandas as pd
    vocabf={}
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(g_model_folder+"tokenizer_bpe_form14.json")
    for k,v in tokenizer.get_vocab().items():
        vocabf[k]=0
    for ltxt in batch_iterator():
        r=tokenizer.encode_batch(ltxt)
        # [Encoding(num_tokens=2, attributes=[ids, type_ids, tokens, offsets,
        # attention_mask, special_tokens_mask, overflowing])]
        for rloc in r:
            tokloc=rloc.tokens
            for tok in tokloc:
                vocabf[tok]+=1
    svocab=pd.Series(vocabf)
    svocab=svocab.sort_values()
    print(svocab.tail(40))
    import pdb;pdb.set_trace()


def train_tokenizer_wp():
    # [00:02:53] for 100 forms
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.WordPieceTrainer(
        vocab_size=20000,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    )
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    #p tokenizer.get_vocab()
    #e=tokenizer.encode('First they generate a contextual embedding for each input token by an autoencoder.')
    #print(e.tokens)
    #p tokenizer.decode(e.ids)
    import pdb;pdb.set_trace()
def get_tokenizer():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(g_model_folder+"tokenizer_bpe_form14.json")
    return tokenizer
# ipython -i -m IdxSEC.corpus_create_v1
if __name__=='__main__':
    #t=train_tokenizer_bpe()
    get_vocab_frequency()
    #t=





