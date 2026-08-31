#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation"""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation"""

    def __init__(self):
        """
        Class constructor
        Sets the instance attributes:
            data_train - the ted_hrlr_translate/pt_to_en train split,
                loaded via load_pt2en('train'), as a tf.data.Dataset
            data_valid - the ted_hrlr_translate/pt_to_en validation split,
                loaded via load_pt2en('validation'), as a tf.data.Dataset
            tokenizer_pt - the Portuguese tokenizer created from the
                training set
            tokenizer_en - the English tokenizer created from the
                training set
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        data is a tf.data.Dataset whose examples are formatted as a
            tuple (pt, en)
            pt is the tf.Tensor containing the Portuguese sentence
            en is the tf.Tensor containing the corresponding English
                sentence
        Uses the pretrained model neuralmind/bert-base-portuguese-cased
            for the portuguese text and the pretrained model
            bert-base-uncased for the english text
        Trains the tokenizers with a maximum vocabulary size of 2 ** 13
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer
        """
        base_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))
        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13
        )
        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English
            sentence
        The tokenized sentences include the start and end of sentence
            tokens
        The start token is indexed as vocab_size
        The end token is indexed as vocab_size + 1
        Returns: pt_tokens, en_tokens
            pt_tokens is a list containing the Portuguese tokens
            en_tokens is a list containing the English tokens
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_ids = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False
        )
        en_ids = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False
        )

        pt_tokens = [pt_vocab_size] + pt_ids + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_ids + [en_vocab_size + 1]

        return pt_tokens, en_tokens
