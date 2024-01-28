import numpy as np
from pprint import pprint
import nlp
import random
import datasets
import os
from pytorch_lightning.loggers import WandbLogger
import torch
import transformers as tfs
import pytorch_lightning as pl
from pytorch_lightning import Trainer
os.environ['TOKENIZERS_PARALLELISM']='false'
# the module nlp is no longer maintened. You should use datasets.
# https://github.com/tshrjn/Finetune-QA/blob/master/data.py
# this seems to run

def seed_everything(seed):
    '''
    Seeds all the libraries for reproducability
    :param int seed: Seed
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class QAModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = tfs.AutoModelForQuestionAnswering.from_pretrained(hparams['qa_model'])

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        return {'loss': loss, 'log': {'train_loss': loss}}

    @property
    def hparams(self):
        return self._hparams

    @hparams.setter
    def hparams(self, value):
        self._hparams = value

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self):
        pass

tokenizer = None
def prepare_data(args):
    #Temporary fix for pl issue #2036
    global tokenizer
    tokenizer = tfs.AutoTokenizer.from_pretrained(args.qa_model, use_fast=True)
    print(tokenizer)

    def _prepare_ds(split):
        #ds = nlp.load_dataset('squad',split='train')
        ds = datasets.load_dataset('squad')[split]
        # default len is 8k
        ds=ds.select([i for i in range(200)])
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()
        #split=f'{split}[:{args.bs if args.fast_dev_run else f"{args.percent}%"}]')

        # ds.cleanup_cache_files()
        ds = ds.map(convert_to_features, batched=True, batch_size=10)

        columns_to_return = ['input_ids', 'token_type_ids', 'attention_mask',
                            'start_positions', 'end_positions']
        ds.set_format(type='torch', columns=columns_to_return)
        dl = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=3) # num_workers is important here
        return dl

    train_dl, valid_dl, test_dl = map(_prepare_ds, ('train')), None, None
    # train_dl, valid_dl, test_dl = map(_prepare_ds, ('train', 'validation', 'test'))
    train_dl = _prepare_ds('train')

    return train_dl, valid_dl, test_dl


def get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text'][0]
    start_idx = answer['answer_start'][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1, end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = list(zip(example_batch['context'], example_batch['question']))
    encodings = tokenizer.batch_encode_plus(input_pairs, pad_to_max_length=True, return_token_type_ids=True)

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    start_positions, end_positions = [], []
    for i, (context, answer) in enumerate(zip(example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_correct_alignement(context, answer)
        start_positions.append(encodings.char_to_token(i, start_idx))
        end_positions.append(encodings.char_to_token(i, end_idx-1))
    encodings.update({'start_positions': start_positions,
                    'end_positions': end_positions})

    return encodings

def experiment(args):
    #seed_everything(seed=args.seed)
    # Namespace(bs=512, lr=0.0001, percent=100, qa_model='deepset/roberta-base-squad2', seed=777, tags=[], workers=4, logger=<pytorch_lightning.loggers.wandb.WandbLogger object at 0xffff394e4f90>)
    #qa_model = QAModel(hparams=args)
    #qa_model = QAModel(hparams={'bs':512,'lr':1e-4,'workers':4,'qa_model':'deepset/roberta-base-squad2'})
    # Since you mentioned PyTorch, the chances are that your process is killed due to "Out of Memory".
    # To resolve this, reduce your batch size till you no longer see the error.
    qa_model = QAModel(hparams={'bs': 10, 'lr': 1e-4, 'workers': 1, 'qa_model': 'deepset/tinyroberta-squad2'})
    train_dl, valid_dl, test_dl = prepare_data(args)

    wandb_logger = WandbLogger(project='qa', offline=True)# tags=args.tags
    wandb_logger.watch(qa_model, log='all')
    args.logger = wandb_logger
    #import pdb;pdb.set_trace()
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/1.5.9/api/pytorch_lightning.trainer.trainer.html?highlight=trainer#module-pytorch_lightning.trainer.trainer
    trainer = pl.Trainer(max_epochs=20,
                         #accelerator='cpu',
                         #devices=3,
                         log_every_n_steps=1,
                         logger=wandb_logger
                         )
    #
    #.from_argparse_args(args)
    print('Trainer fit')
    trainer.fit(qa_model, train_dl, valid_dl)
    #trainer.fit(model, train_dataloader, val_dataloader)
    print('Trainer test')
    trainer.test(qa_model, test_dl)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='QA')
    #parser = pl.Trainer.add_argparse_args(parser)  # Adds all pl's trainer args (like max_epochs)

    optim_args = parser.add_argument_group('Optimization related arguments')
    optim_args.add_argument('--bs', type=int,  default=10, help='Batch Size')
    optim_args.add_argument('--lr', type=float,  default=1e-4, help='Initial Learning rate')

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--percent', type=int, default=100, help='Data% to train')

    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--qa_model', type=str, default='deepset/roberta-base-squad2',
                            help='Model name')
    # Model Choices:
    # QA layer pre-finetuned: https://huggingface.co/models?filter=pytorch,question-answering&search=bert
    # Commonly used: twmkn9/distilroberta-base-squad2, deepset/roberta-base-squad2
    # Finetune QA layer from scratch: roberta-base, roberta-large, distilroberta-base, distilroberta-base


    misc_args = parser.add_argument_group('Logging related & Misc arguments')
    misc_args.add_argument('--seed', type=int, default=777, help='Random Seed')
    misc_args.add_argument('-t','--tags', nargs='+', default=[],help='W&B Tags to associate with run')
    misc_args.add_argument('--workers', type=int, default=1, help='Number of parallel worker threads')

    args = parser.parse_args()
    return args

def test_prepare_data():
    import argparse
    parser = argparse.ArgumentParser(description='QA Data')
    parser.add_argument('--qa_model', type=str, default='distilroberta-base', help='Model name')

    args = parser.parse_args()
    args.workers, args.percent, args.bs, args.fast_dev_run = 10, 100, 100, True
    train_dl, valid_dl, test_dl= prepare_data(args)
    print('Data has been prepared !')

# ipython -i -m IdxSEC.finetune_bert_qa_v2
if __name__ == '__main__':
    args = get_args()
    pprint(vars(args))
    experiment(args)
