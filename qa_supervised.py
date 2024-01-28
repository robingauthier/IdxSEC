import pandas as pd
from transformers import pipeline
import distance
import numpy as np

print('reminder that this has to run in limactl start then lima')
print('Models are saved on lima /home/sachadrevet.linux/.cache/huggingface/hub')
# https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt
# for validation there are multiple answers possible in squad
# pour creer plus de samples utiliser une NER et remplacer les lieux et people par d'autres
dq={
    'q1':'Does the text mention a class A and class B ? Reply by Yes or No',
    'q7':'How many voting rights does the class A share has ? ',
    'q8':'How many voting rights does the class A share has ? ',
}
mydata=[
    {'c':'Applicable percentage ownership is based on 5,943,457,010 shares of Class A common stock '
    'and 882,702,042 shares of Class B common stock outstanding at April 4, 2023. '
    'In computing the number of shares of Class A and Class B common stock beneficially owned by a person and the'
    ' percentage ownership of that person, we'
    ' deemed outstanding shares of Class A common stock subject to options held by that person that are currently'
    ' exercisable within sixty days of April 4, 2023 to be outstanding, ignoring the withholding of shares of '
    'common stock to cover applicable taxes. We did not deem these shares outstanding, however, for the purpose '
    'of computing the percentage ownership of any other person. Beneficial ownership representing less than one '
    'percent is denoted with an asterisk (*).',
     'q1':'Does the text mention a class A and class B ?',
     'a1':'Yes',
     'q2':'How many shares outstanding of Class A stock is there ?',
     'a2':'5,943,457,010',
     'q3':'How many shares outstanding of Class B stock is there ?',
     'a3':'882,702,042',
     'q4': 'What is the date mentioned for the number of shares outstanding ?',
     'a4': 'April 4, 2023',
     },
    {'c':
    'Sergey Brin has 368,712,520 Class B Common Stock and 0 Class A Common Stock. ' 
    'Includes (i) 172,700 shares of Class B common stock held by SMB Pacific 2021 Charitable Remainder '
    'Unitrust I, of which Sergey is the sole trustee; and (ii) 172,700 shares of Class B common stock held by '
    'SMB Pacific 2021 Charitable Remainder Unitrust II, of which Sergey is the sole trustee. The address for SMB '
    'Pacific 2021 Charitable Remainder Unitrust I and SMB Pacific 2021 Charitable Remainder Unitrust II is 555 '
    'Bryant Street, #376, Palo Alto, California 94301.',
    'q1':'Does the text mention a class A and class B ?',
    'a1':'Yes',
    'q2':'Who is the main owner we are refering to here?',
    'a2':'Sergey Brin',
    'a2-1':'Sergey',
    'q3':'How many Restricted Units Stocks ( or RSU) does Sergey has ?',
    'a3':'0'
     },
    {'c':'John L. Hennessy has 33,160 shares of Class A common stock. '
         'Consists of 33,160 shares of Class A common stock held by the Hennessy 1993 Revocable Trust. '
         'John is a trustee of the Hennessy 1993 Revocable Trust and has voting and investment authority '
         'over the shares held by the Trust. The address for the Hennessy 1993 Revocable Trust is 580 '
         'Lomita Drive, Stanford, California 94305.',
     'q1':'How many shares of Class A stock does the person mentioned has ?',
     'a1':'33,160',
     'q2': 'Who is the main owner we are refering to here?',
     'a2': 'John L. Hennessy',
     'a2-1': 'John',
     'q3': 'Who are we talking about ?',
     'a3': 'John L. Hennessy',
     },
    {'c': 'BlackRock has 416,003,093 shares of Class A common stock and 0 shares of Class B common stock. '
          'Based on the most recently available Schedule 13G/A filed with the SEC on February 1, 2023 by BlackRock,'
          ' Inc. BlackRock, Inc., a parent holding company through certain of its subsidiaries, beneficially owned '
          '416,003,093 shares of Class A common stock with sole voting power over 370,294,917 shares and sole '
          'dispositive power over 416,003,093 shares. The address for BlackRock, Inc. is 55 East 52nd Street, '
          'New York, New York 10055.',
     'q2': 'Who is the main owner we are refering to here?',
     'a2': 'BlackRock',
     'q3':'How many Restricted Units Stocks ( or RSU) is there ?',
     'a3':'0',
     'q1': 'How many Exercisable Options  is there ?',
     'a1': '0',
     'q7':'How many voting rights does the class A common stock has ? ',
     'a7':'0',
     'q8': 'How many voting rights does the class B common stock has ? ',
     'a8':'0',
     'q4': 'What is the date mentioned for the number of shares owned ?',
     'a4': 'February 1, 2023',
     'q5': 'When were those numbers published ?',
     'a5': 'February 1, 2023',
     },
    {
    'c':'All executive officers and directors as a group (12 individuals) have 11,656,602 shares.'
        'With respect to the executive officers and directors as a group, consists of an aggregate of '
        '(i) 6,936,532 shares held, (ii) options to purchase 2,440,728 shares of our common stock '
        'exercisable within 60 days of April 19, 2023, (iii) stock awards of 163,092 vesting within '
        '60 days of April 19, 2023, and (iv) 2,116,250 shares exercisable pursuant to the Auerbach Warrant.',
    'q1':'How many individuals have been aggregated here ?',
    'a1':'12 individuals',
    'q2':'How many options to purchase shares is there ?',
    'a2':'2,440,728',
    'q3':'How many shares are held currently ?',
    'a3':'6,936,532',
    }
]
def get_size_of_models():
    txt='''
    456M	./models--jkgrad--xlnet-base-squadv2
    315M	./models--Jayaprakash-JPVL--qa_model_finance_domain_fiqa_v6
    1.3G	./models--deepset--bert-large-uncased-whole-word-masking-squad2
    257M	./models--distilbert-base-uncased
    417M	./models--haddadalwi--multi-qa-mpnet-base-dot-v1-finetuned-squad2-all
    417M	./models--anablasi--qa_financial_v2
    1.3G	./models--bert-large-uncased-whole-word-masking-finetuned-squad
    824M	./models--microsoft--DialoGPT-medium
    696M	./models--facebook--blenderbot-400M-distill
    148K	./models--microsoft--phi-2
    414M	./models--dslim--bert-base-NER
    514M	./models--microsoft--markuplm-base-finetuned-websrc
    475M	./models--deepset--roberta-base-squad2
    476M	./models--Rakib--roberta-base-on-cuad
    250M	./models--distilbert-base-cased-distilled-squad
    337M	./models--microsoft--DialoGPT-small
    2.7M	./models--distilroberta-base
    '''
    print(txt)

def main(model="timpal0l/mdeberta-v3-base-squad2",debug=1):
    """
    model="distilbert-base-cased-distilled-squad"
    model="anablasi/qa_financial_v2"
    model="Jayaprakash-JPVL/qa_model_finance_domain_fiqa_v6"
    model="haddadalwi/multi-qa-mpnet-base-dot-v1-finetuned-squad2-all"
    model="Rakib/roberta-base-on-cuad"
    model="deepset/roberta-base-squad2" ---> the best one
    model="microsoft/markuplm-base-finetuned-websrc"
    model="deepset/bert-large-uncased-whole-word-masking-squad2" --> decent
    model="jkgrad/xlnet-base-squadv2"
    model="microsoft/markuplm-base-finetuned-websrc"--> not working
    model="timpal0l/mdeberta-v3-base-squad2" --> to test
    """
    #
    #qa_pipeline = pipeline("question-answering",model="distilbert-base-cased-distilled-squad")
    start_time=pd.to_datetime('now')
    qa_pipeline = pipeline("question-answering", model=model)
    lr=[]
    for dataloc in mydata:
        context=dataloc['c']
        #context +=' You can reply to a question by Yes or No.'
        context+=('If the answer to the question is yes or no, '
                  'you can reply by Yes I think this is correct or alternatively '
                  'No I do not think this is correct.')
        context+=('If the answer to the question is 0 then you should reply 0.')
        if debug > 0:
            print('-' * 10)
            print('-' * 10)
            print('C: '+context)
        list_q = [k for k in dataloc.keys() if k.startswith('q')]
        for q in list_q:
            question=dataloc[q]
            answer = dataloc[q.replace('q','a')]
            ml_answer = qa_pipeline(question=question, context=context)
            ml_score=ml_answer['score']
            if ml_score<0.03:
                ml_answer_txt='0'
            else:
                ml_answer_txt = ml_answer['answer']
            dst=distance.jaccard(ml_answer_txt,answer)
            lr+=[dst]
            if debug>0:
                print('Q: '+str(question))
                print('ML_Score: %.5f'%ml_answer['score'])
                print('ML_A: '+str(ml_answer['answer']))
                print('A: '+str(answer))
                print('distance : %.4f'%dst)
                print('-'*10)
    avg_dst=np.mean(lr)
    end_time = pd.to_datetime('now')
    print('Model in use:'+model)
    print('Total Average Score : %.3f'%avg_dst)
    print('Total time :'+str(end_time-start_time))
    return avg_dst

def test_evaluate():
    import evaluate
    predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
    references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

def test_adhoc():
    #qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2")
    model='bert-large-uncased-whole-word-masking-finetuned-squad'
    model='distilbert-base-uncased'
    qa_pipeline = pipeline("question-answering", model=model)
    context = (
    'Applicable percentage ownership is based on 5,943,457,010 shares of Class A common stock '
    'and 882,702,042 shares of Class B common stock outstanding at April 4, 2023. '
    'In computing the number of shares of Class A and Class B common stock beneficially owned by a person and the'
    ' percentage ownership of that person, we'
    ' deemed outstanding shares of Class A common stock subject to options held by that person that are currently'
    ' exercisable within sixty days of April 4, 2023 to be outstanding, ignoring the withholding of shares of '
    'common stock to cover applicable taxes. We did not deem these shares outstanding, however, for the purpose '
    'of computing the percentage ownership of any other person. Beneficial ownership representing less than one '
    'percent is denoted with an asterisk (*).'
    'If the answer to the question is yes or no, you can reply by Yes I think this is correct or alternatively '
    'No I do not think this is correct.'
    )
    question = ('Does the text say at some point that there is a class A common stock and a '
                'class B common stock ? When there is no class A and class B the text only refers to '
                'common stock without specifying if those are A or B shares.')
    # pour le coup des grep de class A ou class B peuvent aider ici
    result = qa_pipeline(question=question, context=context)

    # Print the result
    print(f"Question: {question}")
    print(f"Score: {result['score']}")
    print(f"Answer: {result['answer']}")
    print(f"Start index: {result['start']}, End index: {result['end']}")

def test_conversational1():
    from transformers import Conversation
    # model="facebook/blenderbot-400M-distill"
    # , model="microsoft/DialoGPT-small"
    # default model is microsoft/DialoGPT-medium
    conversation_pipeline = pipeline("conversational",model="microsoft/DialoGPT-small")

    conv1 = Conversation("You are a helpful assistant.")
    conv2 = Conversation("Who won the world series in 2020?")

    # Generate model response
    response =conversation_pipeline([conv1, conv2])
    print(response)
    import pdb;pdb.set_trace()
    #print("User:", user_input)
    #print("Model:", response[0]["generated_responses"][-1]["response"])

def test_conversational2():
    from transformers import Conversation
    # model="facebook/blenderbot-400M-distill"
    # , model="microsoft/DialoGPT-small"
    # default model is microsoft/DialoGPT-medium
    conversation_pipeline = pipeline("conversational",model="microsoft/DialoGPT-small")

    conv1 = Conversation(mydata[0]['c']+' '+mydata[0]['q1'])
    conv2 = Conversation(mydata[0]['q2'])
    conv3 = Conversation(mydata[0]['q3'])
    conv4 = Conversation(mydata[0]['q4'])

    # Generate model response
    response =conversation_pipeline([conv1,conv2,conv3,conv4])
    print(response)


def test_squad():
    from datasets import load_dataset

    # Load the SQuAD dataset
    squad_dataset = load_dataset("squad")

    # Print information about the dataset
    print(squad_dataset)
    print(squad_dataset['train'][100])
    import pdb;pdb.set_trace()

def test_qa():
    """
    Important to understand that the QA model predicts the start token of the answer and the end token of the answer
    """
    # https://towardsdatascience.com/simple-and-fast-question-answering-system-using-huggingface-distilbert-single-batch-inference-bcf5a5749571
    from transformers import AutoModelForQuestionAnswering
    from transformers import AutoTokenizer
    from datasets import load_dataset
    import torch
    squad_dataset = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    model=AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
    dataloc=squad_dataset['train'][1000]
    context=dataloc['context']
    question=dataloc['question']
    n0=len(context.split(' '))+len(question.split(' '))
    encoding = tokenizer.encode_plus(question, context, return_tensors="pt")
    # a dictionary of 2 tensors
    # encoding() returns only a list of numbers
    input_ids=encoding["input_ids"]
    n1=len(input_ids[0])
    attention_mask = encoding["attention_mask"]
    qaout = model(input_ids, attention_mask=attention_mask)
    start_scores=qaout['start_logits']
    assert len(start_scores[0])==n1,'issue'
    end_scores= qaout['end_logits']
    ans_tokens = input_ids[0][torch.argmax(start_scores): torch.argmax(end_scores) + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
    print("\nQuestion ", question)
    print("\nAnswer Tokens: ")
    print(answer_tokens)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    print("\nAnswer : ", answer_tokens_to_string)


def test_qa2():
    # Bloom has a size of 9G btw.. it is too big I did not manage to download it
    # TODO: Bloom are Text Generation models mainly... hence we need to try them

    from transformers import BloomForQuestionAnswering
    from transformers import BloomTokenizerFast
    from transformers import AutoTokenizer
    from datasets import load_dataset
    squad_dataset = load_dataset("squad")

    # bigscience/bloom-560m
    #tokenizer = BloomTokenizerFast()
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
    model=BloomForQuestionAnswering.from_pretrained("bigscience/bloom")

    dataloc=squad_dataset['train'][1000]
    context=dataloc['context']
    question=dataloc['question']
    import pdb;pdb.set_trace()

def test_qa3():
    # size of the model is 5G
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model="microsoft/phi-2" # size > 5G
    model='gpt2'
    model = AutoModelForCausalLM.from_pretrained(model,trust_remote_code=True,torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    inputs = tokenizer('''def print_prime(n):
       """
       Print all primes between 1 and n
       """''', return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=200)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

def test_qa4():
    # TFGPT2Model ==> TF means tensorflow
    # https://huggingface.co/gpt2
    # This is the smallest version of GPT-2, with 124M parameters.
    from transformers import pipeline
    from pprint import pprint
    generator = pipeline('text-generation', model='gpt2')
    # GPT-2 va completer le text en fait...
    text = 'Sergey Brin has 368,712,520 Class B Common Stock and 0 Class A Common Stock. How many class B stock has Sergey ?'
    outputs = generator(text)#, max_length=30, num_return_sequences=5,truncation=True
    pprint(outputs)
    # Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    # [{'generated_text': 'Sergey Brin has 368,712,520 Class B Common Stock and 0 '
    #                     'Class A Common Stock. How many class B stock has Sergey ? '
    #                     'Is he?\n'
    #                     '\n'
    #                     'Answer by Dr. Mark Stahlberg, a professor of financial '
    #                     'services at the'}]
    import pdb;pdb.set_trace()


def test_qa5():
    from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline
    import pandas as pd

    table = pd.read_excel("excel path", sheet_name="Data")
    table = table.astype(str)

    model = 'google/tapas-base-finetuned-wtq'
    tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model)
    tapas_tokenizer = AutoTokenizer.from_pretrained(model)

    # Initializing pipeline
    nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

    def qa(query,data):
        print(query)
        result = nlp({'table': data,'query':query})
        answer = result['cells']
        print(answer)

    prediction = qa("What are the different incident types", table)

# ipython -i -m IdxSEC.qa_supervised
if __name__=='__main__':
    #main()
    #test_adhoc()
    test_squad()
    #test_conversational2()
    #test_qa2()
    #test_qa3()
    #test_qa4()