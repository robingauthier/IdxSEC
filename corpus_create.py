from .edgar_utils import get_forms
from .blocks_html_v1 import blocks_html
from . import g_edgar_folder


# https://github.com/huggingface/tokenizers/tree/main
# https://huggingface.co/docs/tokenizers/pipeline
# https://huggingface.co/docs/tokenizers/training_from_memory

def get_form_txt(fname):
    lr = []
    lstruct, _ = blocks_html(fname)
    for elem in lstruct:
        if elem['tag'] != 'table':
            lr += [elem['content']]
    return lr
def batch_iterator_old(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]
def batch_iterator(batch_size=1000):
    lfiles = get_forms()
    for fname in lfiles:
        yield get_form_txt(fname)


def train_tokenizer_old():
    # https://github.com/huggingface/tokenizers/tree/main
    # https://huggingface.co/docs/tokenizers/pipeline
    # https://huggingface.co/docs/tokenizers/training_from_memory
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    from tokenizers.processors import BertProcessing
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=20000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    )
    data = [
        "Beautiful is better than ugly."
        "Explicit is better than implicit."
        "Simple is better than complex."
        "Complex is better than complicated."
        "Flat is better than nested."
        "Sparse is better than dense."
        "Readability counts."
    ]
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2,
                    special_tokens=[
                        "<s>", ])
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )



def train_tokenizer():

    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=20000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    )
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    import pdb;pdb.set_trace()

# ipython -i -m IdxSEC.corpus_create
if __name__=='__main__':
    t=train_tokenizer()





