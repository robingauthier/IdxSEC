import pandas as pd
import numpy as np
from pprint import pprint
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from functools import partial
import torch
# here you need to carefully read this : https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt

def get_answer_position(context, answer):
    """ context is a string, answer as well
    we return the character index of the answer
    and 0 if the answer is not found
    """
    start_idx = context.find(answer)
    if start_idx==-1:
        return 0,0
    end_idx = start_idx + len(answer)
    return start_idx, end_idx

def preprocess_data():
    """transforms mydata dict into the squad format
    since we have multiple questions by context we need to convert
    into 1 question for 1 context.

    """
    from .data_qa import mydata
    ndata=[]
    missing_answer=[]
    for dataloc in mydata:
        qestionks = [x for x in dataloc.keys() if x.startswith('q')]
        context=dataloc['c']
        for questionk in qestionks:
            answerk = questionk.replace('q','a')
            resloc={}
            resloc['context']=context
            resloc['question']=dataloc[questionk]
            resloc['answers']=dataloc[answerk] # Caution name is answers
            ndata+=[resloc]
            check_answer_exists=get_answer_position(context, resloc['answers'])
            missing_answer+=[1*(check_answer_exists==0)]
    if pd.Series(missing_answer).mean()>0.3:
        print('Warning many missing answers in preprocess_data')
    return ndata

def convert_to_features(example_batch,tokenizer,debug=0):
    """
    when sending with batched=True
     example_batch['question'] is the list of the batch_size questions
     example_batch['context'] is the list of the batch_size contexts

    l'attention_mask c'est uniquement pour le padding
    pour dire ou est la question, c'est le token_type_ids qui ne marche pas sur tous les encoders

    ensuite tout l'enjeu c'est de calculer les features : start_positions et end_positions
    car ils doivent correspondre a la position du token ou commence la reponse et ou elle finit.
    donc il faut convertir la position du charactere dans le string en une position du token dans input_ids

    all the tokenizers are not the same and do not responds the same way to arguments in tokenizer()
    """
    batch_size=len(example_batch['context'])

    # Tokenize contexts and questions (as pairs of inputs)
    input_pairs = list(zip(example_batch['question'],example_batch['context']))
    encodings = tokenizer(input_pairs, return_tensors="pt",
                          padding='max_length',
                          max_length=2**8,
                          truncation=True,
                          return_token_type_ids=True,
                          return_offsets_mapping=True)

    if np.unique(encodings['token_type_ids'][0].numpy()).shape[0]==1:
        print('Caution the token_type_ids did not work at all')
        print(encodings['token_type_ids'][0])

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    not_found_position=0
    start_positions, end_positions = [], []
    for i, (question,context, answer) in enumerate(zip(example_batch['question'],example_batch['context'], example_batch['answers'])):
        start_idx, end_idx = get_answer_position(context, answer)

        if start_idx==0:
            start_positions.append(not_found_position)
            end_positions.append(not_found_position)
            continue

        # now we need to find the token position
        # how does posdf look like ?
        # 0    0   0 # CLS token
        # 1    0   3 # this are question characters
        # 11  54  55  # end of question characters
        # 12   0   0  # sep token
        # 14   0   6  # start of context characters
        # 15   6  10
        posdf=pd.DataFrame(encodings['offset_mapping'][i].numpy())
        posdf.columns=['start_idx','end_idx']
        posdf['qorc']=encodings.sequence_ids(i)
        posdf=posdf.loc[lambda x:(x['start_idx']>0)&(x['end_idx']>0)]
        posdf = posdf.loc[lambda x: x['qorc'] == 1] # answer is in the context
        posdf=posdf.loc[lambda x:x['start_idx']>=start_idx]
        posdf=posdf.loc[lambda x: x['start_idx']<=end_idx]
        if posdf.shape[0]==0:
            start_positions.append(not_found_position)
            end_positions.append(not_found_position)
            continue

        start_token_idx=posdf.index[0]
        end_token_idx = posdf.index[-1]+1

        if debug>0:
            print('-'*10)
            pprint({
                'answer_real':answer,
                'check_start_idx':context[start_idx:end_idx],
                'check_token_idx':tokenizer.decode(encodings['input_ids'][i][start_token_idx:end_token_idx]),
                 })

        start_positions.append(start_token_idx)
        end_positions.append(end_token_idx)

    assert len(start_positions)==batch_size,'issue on len'
    encodings.update({'start_positions': torch.tensor(start_positions,dtype=torch.long),
                    'end_positions': torch.tensor(end_positions,dtype=torch.long)})
    return encodings



def main(debug=0):
    from transformers import pipeline
    ds0=preprocess_data()
    ds = Dataset.from_list(ds0)
    model='deepset/roberta-base-squad2'
    model='deepset/bert-large-uncased-whole-word-masking-squad2'
    tokenizer = AutoTokenizer.from_pretrained(model)
    model=AutoModelForQuestionAnswering.from_pretrained(model)

    convert_to_features_with_tokenizer = partial(convert_to_features, tokenizer=tokenizer)

    ds = ds.map(convert_to_features_with_tokenizer, batched=True, batch_size=10)
    #ds = ds.map(convert_to_features_with_tokenizer, batched=False)
    columns_to_return = ['input_ids', 'attention_mask','start_positions', 'end_positions','token_type_ids']
    ds.set_format(type='torch', columns=columns_to_return)
    ds.save_to_disk('/Users/sachadrevet/src/IdxSEC/model/data-arrow')
    #dl = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=3)


    # checks on the size of the data
    sdf=pd.DataFrame({'input_ids':[x['input_ids'].shape[0] for x in ds],
                      'attention_mask':[x['attention_mask'].shape[0] for x in ds],
                      })
    print(sdf)
    assert len(sdf['input_ids'].unique())==1,'issue not unique length'

    # deactivate gradient on most of the model
    for name,param in model.named_parameters():
        if name.startswith('qa_outputs'):
            param.requires_grad=True
        else:
            param.requires_grad=False
        if debug>0:
            print(name)
            print(param.shape)
            print(param.requires_grad)

    from torch.optim import AdamW
    from transformers import TrainingArguments, Trainer, logging
    from torchinfo import summary
    training_args = TrainingArguments(
        output_dir='/Users/sachadrevet/src/IdxSEC/model/',
        overwrite_output_dir=True,
        report_to="none" # disables wandb
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    rtrain = trainer.train()

    # TrainOutput(global_step=9, training_loss=3.32831170823839,
    # metrics={'train_runtime': 77.9514, 'train_samples_per_second': 0.77,
    # 'train_steps_per_second': 0.115, 'train_loss': 3.32831170823839, 'epoch': 3.0})

    # Saving the model is too hard for my laptop because it weights>1G
    #trainer.model.save_pretrained('/Users/sachadrevet/src/IdxSEC/model/sacha-qa-20230122-pytorch')
    #predictions, _, _ = trainer.predict(ds)
    # predictions is a list of start_token,end_token logits
    # start_logits, end_logits = predictions
    # compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
    # this is crashing the python as well
    #trainer.save_model('/Users/sachadrevet/src/IdxSEC/model/sacha-qa-20230122')
    #nmodel = AutoModelForQuestionAnswering.from_pretrained(save_directory)

    nlp_qa = pipeline("question-answering", model=trainer.model, tokenizer=tokenizer)
    r=nlp_qa('how old I am ? ','I am 42 years old')
    print(r)
    from .data_qa import mydata
    r=nlp_qa(mydata[0]['q2'],mydata[0]['c'])
    print(r)
    import pdb;pdb.set_trace()

# ipython -i -m IdxSEC.pytorch.prepare_data_qa_v1
if __name__=='__main__':
    main()
    #with open('/Users/sachadrevet/src/IdxSEC/model/test','wt') as f:
    #    f.write('helo')
