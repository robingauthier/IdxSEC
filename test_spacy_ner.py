import spacy
import re

# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# https://huggingface.co/dslim/bert-base-NER

def extract_information(sentence):
    doc = nlp(sentence)
    information = {'name': None, 'shares': None, 'options': None}

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            information['name'] = ent.text
        elif ent.text.lower() == 'shares':
            # Use regex to extract numerical values after "shares"
            matches = re.findall(r'\b\d+\b', sentence)
            if matches:
                information['shares'] = int(matches[0])
        elif ent.text.lower() == 'options':
            # Use regex to extract numerical values after "options"
            matches = re.findall(r'\b\d+\b', sentence)
            if matches:
                information['options'] = int(matches[0])

    return information

# ipython -i -m IdxSEC.test_spacy_ner
if __name__=='__main__':
    # Example sentence
    sentence = "Consists of (i) 26,476 shares held by Mr. Ludwig, and (ii) options to purchase 381,769 shares of our common stock exercisable within 60 days of April 19, 2023."
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.label_)
        print(ent.text)
        print('-'*20)
    # Extract information
    #result = extract_information(sentence)

    # Display the result
    #print(result)
