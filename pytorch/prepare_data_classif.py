import pandas as pd
import numpy as np
from pprint import pprint
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig
from transformers import DataCollatorWithPadding
from functools import partial
import torch
import evaluate

from transformers import pipeline

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# ipython -i -m IdxSEC.pytorch.prepare_data_classif
if __name__=='__main__':
    from .data_classif import mydata
    ds = Dataset.from_list(mydata)

    model = 'deepset/roberta-base-squad2'
    model = 'deepset/bert-large-uncased-whole-word-masking-squad2'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(
        model,
        num_labels=6,
        id2label={0: 'other', 1: 'ownership',2:'2',3:'3',4:'4',5:'5'})
    #config = AutoConfig.from_pretrained(model)
    metric = evaluate.load("accuracy")

    # changing the number of labels
    #config.num_labels = 6
    #model = AutoModelForSequenceClassification.from_config(config)
    #import pdb;pdb.set_trace()

    def tokenize_function(examples):
        """ca rajoute les features input_ids,attention_mask aux features existants"""
        return tokenizer(examples["s"], padding="max_length",max_length=2**8,return_tensors="pt", truncation=True)
    ds = ds.map(tokenize_function, batched=True)
    ds.remove_columns(['s'])
    label_map={'other':0,'ownership':1}
    def convert_label(examples):
        """
        input is
        (Pdb) p examples['label']
        ['other', 'other', 'other', 'ownership', 'ownership', 'ownership', 'other', 'other', 'other', 'ownership', 'ownership', 'other']
        """
        #import pdb;pdb.set_trace()
        return {'label':[torch.tensor(label_map[x]) for x in examples['label']],'olabel':examples['label']}
    ds = ds.map(convert_label, batched=True)
    #data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='longest',return_tensors='pt')


    # deactivate gradient on most of the model
    for name,param in model.named_parameters():
        if name.startswith('classifier'):
            param.requires_grad=True
        else:
            param.requires_grad=False
        if False:
            print(name)
            print(param.shape)
            print(param.requires_grad)

    print(model.classifier.out_features)
    print(model.num_labels)
    print(model.classifier.parameters)

    from transformers import TrainingArguments, Trainer, logging
    from torchinfo import summary
    training_args = TrainingArguments(
        output_dir='/Users/sachadrevet/src/IdxSEC/model/',
        overwrite_output_dir=True,
        report_to="none" # disables wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        #eval_dataset=eval_dataset,
        #data_collator=data_collator
    )
    rtrain = trainer.train()


    nlp_class = pipeline('text-classification', model=trainer.model, tokenizer=tokenizer)
    r=nlp_class('I am 42 years old')
    print(r)
    from .data_classif import mydata
    r=nlp_class(mydata[0]['s'])
    print(r)
    #import pdb;pdb.set_trace()