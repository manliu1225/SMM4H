# coding=utf-8
"""BERT embedding."""
from typing import List

import gluonnlp
import mxnet as mx
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from mxnet.gluon.data import DataLoader

from .dataset import BertEmbeddingDataset


class BertEmbedding(object):
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    """

    def __init__(self, ctx=mx.cpu(), model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased',
                 max_seq_length=25, batch_size=256):
        """
        Encoding from BERT model.

        Parameters
        ----------
        ctx : Context.
            running BertEmbedding on which gpu device id.
        model : str, default bert_12_768_12.
            pre-trained BERT model
        dataset_name : str, default book_corpus_wiki_en_uncased.
            pre-trained model dataset
        max_seq_length : int, default 25
            max length of each sequence
        batch_size : int, default 256
            batch size
        """
        self.ctx = ctx
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=dataset_name,
                                                         pretrained=True, ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False)

    def __call__(self, sentences, oov_way='avg'):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        return self.embedding(sentences, oov_way='avg')

    def embedding(self, sentences, oov_way='avg'):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        data_iter = self.data_loader(sentences=sentences)
        batches = []
        for token_ids, valid_length, token_types in data_iter:
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs = self.bert(token_ids, token_types,
                                         valid_length.astype('float32'))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way)

    def data_loader(self, sentences, shuffle=False):
        # tokenizer = BERTTokenizer(self.vocab)
        # transform = BERTSentenceTransform(tokenizer=tokenizer,
        #                                   max_seq_length=self.max_seq_length,
        #                                               pair=False)

        class Listtolist(object):
            def __init__(self, vocab, lower=False, max_input_chars_per_word=200):
                self.vocab = vocab
                self.max_input_chars_per_word = max_input_chars_per_word
            def __call__(self, sample):
                return sample
            def convert_tokens_to_ids(self, tokens):
                """Converts a sequence of tokens into ids using the vocab."""
                return self.vocab.to_indices(tokens)

        tokenizer = Listtolist(self.vocab)
        transform = BERTSentenceTransform(tokenizer=tokenizer,
                                          max_seq_length=self.max_seq_length,
                                          pair=False)

        dataset = BertEmbeddingDataset(sentences, transform)
        # for line in dataset: print(line)
        # print(dataset)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg'):
        """
        How to handle oov

        Parameters
        ----------
        batches : List[(tokens_id,
                        sequence_outputs,
                        pooled_output].
            batch   token_ids (max_seq_length, ),
                    sequence_outputs (max_seq_length, dim, ),
                    pooled_output (dim, )
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    break
                if token_id in (2, 3):
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences
