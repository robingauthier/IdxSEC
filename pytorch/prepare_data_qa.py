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

def test_squad():
    # Load the SQuAD dataset
    squad_dataset = load_dataset("squad")

    # Print information about the dataset
    print(squad_dataset)
    print(squad_dataset['train'][100])

    squad_sample = {'id': '573387acd058e614000b5cb5', 'title': 'University_of_Notre_Dame',
                    'context': 'One of the main driving forces in the growth of the University was its football team, the Notre Dame Fighting Irish. Knute Rockne became head coach in 1918. Under Rockne, the Irish would post a record of 105 wins, 12 losses, and five ties. During his 13 years the Irish won three national championships, had five undefeated seasons, won the Rose Bowl in 1925, and produced players such as George Gipp and the "Four Horsemen". Knute Rockne has the highest winning percentage (.881) in NCAA Division I/FBS football history. Rockne\'s offenses employed the Notre Dame Box and his defenses ran a 7–2–2 scheme. The last game Rockne coached was on December 14, 1930 when he led a group of Notre Dame all-stars against the New York Giants in New York City.',
                    'question': 'In what year did the team lead by Knute Rockne win the Rose Bowl?',
                    'answers': {'text': ['1925'], 'answer_start': [354]}}
    print(squad_sample)

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


def convert_to_features(example_batch,tokenizer):
    """
    when sending with batched=True
     example_batch['question'] is the list of the batch_size questions
     example_batch['context'] is the list of the batch_size contexts

    l'attention_mask c'est uniquement pour le padding
    pour dire ou est la question, c'est

    all the tokenizers are not the same and do not responds the same way to arguments in tokenizer()
    """
    batch_size=len(example_batch['context'])
    # Tokenize contexts and questions (as pairs of inputs)
    #input_pairs = list(zip(example_batch['context'], example_batch['question']))
    input_pairs = list(zip(example_batch['question'],example_batch['context']))
    #input_pairs = [[question, context] for question, context in zip(example_batch['question'], example_batch['context'])]
    #encodings = tokenizer.batch_encode_plus(input_pairs,pad_to_max_length=True, return_token_type_ids=True, return_tensors="pt")
    encodings = tokenizer(input_pairs, return_tensors="pt",
                          padding='max_length',
                          #padding=True,
                          max_length=2**8,
                          truncation=True,
                          #truncation='only_second',
                          #token_type_id=True,
                          return_token_type_ids=True,
                          return_offsets_mapping=True)
    #  token_type_ids contains only 0

    #import pdb;pdb.set_trace()
    #encodings=tokenizer('question','context',return_token_type_ids=True)
    if np.unique(encodings['token_type_ids'][0].numpy()).shape[0]==1:
        print('Caution the token_type_ids did not work at all')
        print(encodings['token_type_ids'][0])
    #assert encodings['input_ids'].shape[0]==batch_size,'issue'
    #encodings = tokenizer('my qestion','my context is', return_tensors="pt", padding='max_length', truncation=True,return_offsets_mapping=True)
    #tokenizer(['hi','you are cute'])
    # encodings['input_ids'].shape == (10,512)
    # tokenizer.decode(encodings['input_ids'][0]) --> here you see the question first
    # encodings['attention_mask'][0] --> c'est pas bon ca

    # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
    not_found_position=0
    start_positions, end_positions = [], []
    #import pdb;pdb.set_trace()
    for i, (question,context, answer) in enumerate(zip(example_batch['question'],example_batch['context'], example_batch['answers'])):
        #print('-'*10)
        #print(i)
        #import pdb;pdb.set_trace()
        question_context = tokenizer.decode(encodings['input_ids'][i])
        question_context = question +' '+context
        # encodings.sequence_ids(i) permet de voir le split question context
        start_idx, end_idx = get_answer_position(context, answer)

        if start_idx==0:
            print('start_idx=0')
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
        # the help on the tokenizers is in the module tokenizers
        # encodings.char_to_token(i, 10) it tells you that character nb 20 is the token nb 5
        #start_res=encodings.char_to_token(start_idx,sequence_index=i)
        #end_res=encodings.char_to_token(end_idx,sequence_index=i)
        #print(tokenizer.decode(encodings['input_ids'][i][encodings.char_to_token(0,sequence_index=i+1)]))
        print('-'*10)
        pprint({
            'answer_real':answer,
            'check_start_idx':context[start_idx:end_idx],
            'check_token_idx':tokenizer.decode(encodings['input_ids'][i][start_token_idx:end_token_idx]),
             })
        #print(tokenizer.decode(encodings['input_ids'][i][0:10]))
        #import pdb;pdb.set_trace()

        start_positions.append(start_token_idx)
        end_positions.append(end_token_idx)
        #print(start_positions)
        #assert len(start_positions)==i+1,'issue len'

    assert len(start_positions)==batch_size,'issue on len'
    encodings.update({'start_positions': torch.tensor(start_positions,dtype=torch.long),
                    'end_positions': torch.tensor(end_positions,dtype=torch.long)})
    return encodings




def test_create_qa_dataset():
    from datasets import Dataset

    data = [
        {'question': 'What is the capital of France?',
         'context': 'Paris is the capital and most populous city of France.',
         'answer': 'Paris',
         'answer_start': 8
         },

        {'question': 'Who is the CEO of Apple?',
         'context': 'Tim Cook is the current CEO of Apple Inc.',
         'answer': 'Tim Cook',
         'answer_start': 0
         }
    ]

    dataset = Dataset.from_dict(data)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']




# ipython -i -m IdxSEC.pytorch.prepare_data_qa
if __name__=='__main__':
    #test_create_qa_dataset()
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
        if True:
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
    trainer.train()
    if False:

        dl = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=3)  # num_workers is important here
