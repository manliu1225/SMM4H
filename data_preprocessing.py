#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing
# 
# generate integer-indexed sentences, pos-tags and named entity tags, dictionaries for converting, etc, and save as `npy` binaries.


import pandas as pd
import numpy as np
from preprocessing import get_vocab, index_sents
from embedding import create_embeddings
from sklearn.model_selection import train_test_split
from bert_embedding.bert import BertEmbedding
import feature_namelist
import re
# set maximum network vocabulary, test set size
MAX_VOCAB = 25000
TEST_SIZE = 0.15


# ### read ConLL2002 NER corpus from csv (first save as utf-8!)


data = pd.read_csv('./data/data.txt', delimiter = "\t")

sentmarks = data["Sentence #"].tolist()
sentmarks = [str(s) for s in sentmarks]
sentmarks_li = []
for i, s in enumerate(sentmarks, 1):
    if s == "nan":
        sentmarks_li.append(s)
    else:
        sentmarks_li.append("Sentence: {}".format(i))

sentmarks = sentmarks_li

words = data["Word"].tolist()
postags = data["POS"].tolist()
nertags = data["Tag"].tolist()

sentence_text = []
sentence_post = []
sentence_ners = []

vocab = []

this_snt = []
this_pos = []
this_ner = []

for idx, s in enumerate(sentmarks):
    if s != 'nan': # the begin of sent
        if len(this_snt) > 0 and this_snt[-1] == '0':
            if list(set(this_ner)) != ['O']:
                sentence_text.append(this_snt[:-1])
                sentence_post.append(this_pos[:-1])
                sentence_ners.append(this_ner[:-1])
        this_snt = []
        this_pos = []
        this_ner = []
    
    # add to lists 
    this_snt.append(words[idx].lower())
    this_pos.append(postags[idx])
    this_ner.append(nertags[idx])
    vocab.append(words[idx].lower())


# ## get vocabulary and index inputs
# 
# we need to convert the string input to integer vectors for the `keras` network (the `pycrfsuite` network needs strings, as it will extract feature vectors from words themselves).
# 
# we will index each word from 1 according to inverse frequency (most common word is 1, etc.) until the max-vocab size. We will reserve two slots, 0 for the PAD index, and MAX_VOCAB-1 for out-of-vocabulary or unknown words (OOV/UNK). Since this is boring stuff, I've put it in external functions. Packages like `keras` and `sklearn` have more robust tools for this, but a simple word:index dictionary will do fine for this experiment


# text vocab dicts
# subtract 2 for UNK, PAD
word2idx, idx2word = get_vocab(sentence_text, MAX_VOCAB-2)


# POS and NER tag vocab dicts
pos2idx, idx2pos = get_vocab(sentence_post, len(set(postags)))
ner2idx, idx2ner = get_vocab(sentence_ners, len(set(nertags))+2)


# print(sentence_post)

# index
sentence_text_idx = index_sents(sentence_text, word2idx)
sentence_post_idx = index_sents(sentence_post, pos2idx)
sentence_ners_idx = index_sents(sentence_ners, ner2idx)


# ## train-test splitting
# 
# we divide the training data into training data, and testing data. the testing data is used only for checking model performance. a third set, the *validation set*, may be split off from our training data for hyperparameter tuning, although if we use k-fold cross-validation, our validation set will change every fold.


indices = [i for i in range(len(sentence_text))]

# print(sentence_post_idx)
train_idx, test_idx, X_train_pos, X_test_pos = train_test_split(indices, sentence_post_idx, test_size=TEST_SIZE)

def get_sublist(lst, indices):
    result = []
    for idx in indices:
        result.append(lst[idx])
    return result

X_train_sents = get_sublist(sentence_text_idx, train_idx)
X_test_sents = get_sublist(sentence_text_idx, test_idx)
y_train_ner = get_sublist(sentence_ners_idx, train_idx)
y_test_ner = get_sublist(sentence_ners_idx, test_idx)


# ## create word2vec embeddings for words, pos-tags
# 
# using pre-trained embedding vectors to initialize the embedding layer has been shown to help training for various sequence labeling tasks such as POS tagging (Huang, Xu & Yu 2015; Ma & Hovy 2016) and Named Entity Recognition for English (Ma & Hovy 2016; Lee Changki 2017) and Japanese (Misawa, Taniguchi, Miura & Ohkuma 2017).
# 
# because we are using the POS-tags as a secondary input, we will also train an embedding space fo these. we will use only the training data to create the embeddings. i am using `gensim` for this task, and i am using a helper function to wrap the `Word2Vec` that saves the embedding and also the vocabulary dictionary.


# sentence embeddings
train_sent_texts = [sentence_text[idx] for idx in train_idx]
# print(train_sent_texts) 

w2v_vocab, w2v_model = create_embeddings(train_sent_texts,
                       embeddings_path='embeddings/text_embeddings.gensimmodel',
                       vocab_path='embeddings/text_mapping.json',
                       size=300,
                       workers=4,
                       iter=20)

# pos embeddings
train_post_texts = [sentence_post[idx] for idx in train_idx]

w2v_pvocab, w2v_pmodel = create_embeddings(train_post_texts,
                         embeddings_path='embeddings/npos_embeddings.gensimmodel',
                         vocab_path='embeddings/npos_mapping.json',
                         size=100,
                         workers=4,
                         iter=20)


# ## save everything to numpy binaries for loading
# 
# granted, `pickle` would probably be more suitable for a lot of these things. but over-reliance on `numpy` binaries is a bad habit i've picked up.
## dictionary features, [None, 30, 3]
## dictionary feature is list of list, then convert to array
print(y_train_ner[0]) # list of list
print([idx2word[e] for e in X_train_sents[0]])
# print(X_test_sents.shape) # (229,)
print(len(sentence_text))
print(len(X_train_sents))



# Namelist features
namelist_filenames = ["namelist_adr.txt", "disease.txt"]
MANUAL_MUSIC_DIR = 'resources/dictionary'
features_dict = {}
features_dict["nameListFeature"] = feature_namelist.load_namelist(
            namelist_filenames, MANUAL_MUSIC_DIR, skip_first_row=True)

namelist_ADR = []
for instance_tokens in sentence_text:
    instance_ADR, instance_O = [], []
    instance_tokens_lower = [re.sub(r'!|\?|\"|\'', '', e.lower()) for e in instance_tokens]
    namelist_idx = feature_namelist.get_namelist_match_idx(
            features_dict["nameListFeature"], instance_tokens_lower) 
    for idx, token in enumerate(instance_tokens):
            if idx in namelist_idx:
                namelist_dict = {"NameList:ADR" : 1, "NameList:O" : 10}
                # print(namelist_idx.get(idx, "")["pos"])
                # start = ""
                start = 2 if namelist_idx.get(idx, "")["pos"] == 0 else 0
                instance_ADR.append(namelist_dict["NameList:ADR"]+start if "ADR" in namelist_idx.get(idx, "")["labels"] else 0)
                instance_O.append(namelist_dict["NameList:O"] if "O" in namelist_idx.get(idx, "")["labels"] else 0)
            else:
                instance_ADR.append(0)
                instance_O.append(0)

    namelist_ADR.append(instance_ADR)

# print(namelist_ADR)

from data_preprocessing import get_sublist
X_train_features = np.array(get_sublist(namelist_ADR, train_idx))
X_test_features = np.array(get_sublist(namelist_ADR, test_idx))
### 


### bert word embedding
X_train_sent_li = [sentence_text[idx] for idx in train_idx]

X_test_sent_li = [sentence_text[idx] for idx in test_idx]
bert_embedding = BertEmbedding(model="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased",
                         max_seq_length=30, batch_size=256)

X_train_sents_bert = [e[1] for e in bert_embedding(X_train_sent_li, oov_way="avg")]
X_test_sents_bert = [e[1] for e in bert_embedding(X_test_sent_li, oov_way="avg")]
print(np.array(X_train_sents_bert).shape)
###


def numpy_save(saves, names):
    for idx, item in enumerate(saves):
        np.save('../encoded/{0}.npy'.format(names[idx]), item)
    return

saves = [
vocab,
sentence_text_idx,
sentence_post_idx,
sentence_ners_idx,
word2idx, idx2word,
pos2idx, idx2pos,
ner2idx, idx2ner,
train_idx,
test_idx,
X_train_sents,
X_test_sents,
X_train_pos,
X_test_pos,
y_train_ner,
y_test_ner,
sentence_text,
sentence_post,
sentence_ners,
X_train_features,
X_test_features,
X_train_sents_bert,
X_test_sents_bert
]

names = [
'vocab',
'sentence_text_idx',
'sentence_post_n_idx',
'sentence_ners_idx',
'word2idx', 'idx2word',
'npos2idx', 'idx2npos',
'ner2idx', 'idx2ner',
'train_idx',
'test_idx',
'X_train_sents',
'X_test_sents',
'X_train_npos',
'X_test_npos',
'y_train_ner',
'y_test_ner',
'sentence_text',
'sentence_npost',
'sentence_ners',
'X_train_features',
'X_test_features',
'X_train_sents_bert',
'X_test_sents_bert']

numpy_save(saves, names)





