#!/usr/bin/env python3
"""
Question Answering module
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question
    using BERT.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/tensorflow/bert_uncased_L-12_H-768_A-12/1')

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + r_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (len(q_tokens) + 2) + [1] * (len(r_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(tf.cast(t, tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    outputs = model(dict(
        input_word_ids=input_word_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids
    ))

    start_logits = outputs[0]
    end_logits = outputs[1]

    ans_start = tf.argmax(start_logits, axis=1)[0].numpy()
    ans_end = tf.argmax(end_logits, axis=1)[0].numpy()

    answer_tokens = tokens[ans_start:ans_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer or ans_start > ans_end or ans_start == 0:
        return None

    return answer
