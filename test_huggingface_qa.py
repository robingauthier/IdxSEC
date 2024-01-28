from transformers import pipeline
from transformers import AutoModelForQuestionAnswering,TFAutoModelForQuestionAnswering


def test_1():
    # No model was supplied, defaulted to distilbert-base-cased-distilled-squad and revision 626af31 (https://huggingface.co/distilbert-base-cased-distilled-squad).
    qa_pipeline = pipeline("question-answering")

    context = (
        "Hugging Face is a company that is focused on natural language processing. "
        "Their Transformers library is widely used for various NLP tasks."
    )

    question = "What is Hugging Face known for?"

    result = qa_pipeline(question=question, context=context)

    # Print the result
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Start index: {result['start']}, End index: {result['end']}")

# ipython -i -m IdxSEC.test_huggingface_qa
if __name__=='__main__':
    # this model is interesting I think
    # https://huggingface.co/datasets/AdaptLLM/finance-tasks?row=51
    print('e')
    test_1()
