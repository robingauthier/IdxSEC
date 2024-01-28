
# https://huggingface.co/dslim/bert-base-NER
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


# https://huggingface.co/datasets/conll2003/viewer/conll2003/train?p=1&row=103
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin. Hello this is a table \n a \t b \t c\t"
print(tokenizer.tokenize(example))
ner_results = nlp(example)
print(ner_results)
# [{'entity': 'B-PER',
#   'score': 0.9990139,
#   'index': 4,
#   'word': 'Wolfgang',
#   'start': 11,
#   'end': 19},
#  {'entity': 'B-LOC',
#   'score': 0.999645,
#   'index': 9,
#   'word': 'Berlin',
#   'start': 34,
#   'end': 40}]

# ipython -i -m IdxSEC.test_huggingface_ner
if __name__=='__main__':
    print('ok')