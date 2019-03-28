#!/usr/bin/env python
# coding: utf-8


# # the archtecture of codes are from  https://github.com/SNUDerek/NER_bLSTM-CRF
# # bidirectional-LSTM-CRF in Keras
# 
# this is a bidirectional LSTM-CRF model for NER, inspired by:
# 
# Huang, Xu, Yu: *Bidirectional LSTM-CRF Models for Sequence Tagging* (2015)
# 
# ...though this is becoming a common architecture for sequence labeling in NLP.

from bert_embedding.bert import BertEmbedding
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import concatenate, Input, LSTM, Dropout, Embedding
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from gensim.models import Word2Vec
from keras_tqdm import TQDMNotebookCallback
from embedding import load_vocab
import feature_namelist
import sys
import re
# ### limit GPU usage for multi-GPU systems
# 
# comment this if using a single GPU or CPU system

# In[2]:


# restrict GPU usage here
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# ## define hyperparameters


# network hyperparameters
MAX_LENGTH = 30
MAX_VOCAB = 25000    # see preprocessing.ipynb
WORDEMBED_SIZE = 300 # see data_preprocessing.ipynb
POS_EMBED_SIZE = 100 # see data_preprocessing.ipynb
HIDDEN_SIZE = 400    # LSTM Nodes/Features/Dimension
BATCH_SIZE = 64
DROPOUTRATE = 0.25
MAX_EPOCHS = 20      # max iterations, early stop condition below



# load data from npys (see preprocessing.ipynb)
print("loading data...\n")
vocab = list(np.load('../encoded/vocab.npy'))
sentence_text = list(np.load('../encoded/sentence_text.npy'))
sentence_post = list(np.load('../encoded/sentence_post.npy'))
sentence_ners = list(np.load('../encoded/sentence_ners.npy'))
sentence_text_idx = np.load('../encoded/sentence_text_idx.npy')
sentence_post_idx = np.load('../encoded/sentence_post_idx.npy')
sentence_ners_idx = np.load('../encoded/sentence_ners_idx.npy')
word2idx = np.load('../encoded/word2idx.npy').item()
idx2word = np.load('../encoded/idx2word.npy').item()
pos2idx = np.load('../encoded/pos2idx.npy').item()
idx2pos = np.load('../encoded/idx2pos.npy').item()
ner2idx = np.load('../encoded/ner2idx.npy').item()
idx2ner = np.load('../encoded/idx2ner.npy').item()
train_idx = np.load('../encoded/train_idx.npy')
test_idx = np.load('../encoded/test_idx.npy')
X_train_sents = np.load('../encoded/X_train_sents.npy')
X_test_sents = np.load('../encoded/X_test_sents.npy')
X_train_pos = np.load('../encoded/X_train_pos.npy')
X_test_pos = np.load('../encoded/X_test_pos.npy')
y_train_ner = np.load('../encoded/y_train_ner.npy')
y_test_ner = np.load('../encoded/y_test_ner.npy')
X_test_features = np.load('../encoded/X_test_features.npy')
X_train_features = np.load('../encoded/X_train_features.npy')
X_train_sents_bert = np.load('../encoded/X_train_sents_bert.npy')
X_test_sents_bert = np.load('../encoded/X_test_sents_bert.npy')


# load embedding data
w2v_vocab, _ = load_vocab('embeddings/text_mapping.json')
w2v_model = Word2Vec.load('embeddings/text_embeddings.gensimmodel')
w2v_pvocab, _ = load_vocab('embeddings/pos_mapping.json')
w2v_pmodel = Word2Vec.load('embeddings/pos_embeddings.gensimmodel')


# ## pad sequences
# 
# we must 'pad' our input and output sequences to a fixed length due to Tensorflow's fixed-graph representation.


# zero-pad the sequences to max length
print("zero-padding sequences...\n")
X_train_sents = sequence.pad_sequences(X_train_sents, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test_sents = sequence.pad_sequences(X_test_sents, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_train_pos = sequence.pad_sequences(X_train_pos, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test_pos = sequence.pad_sequences(X_test_pos, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_train_ner = sequence.pad_sequences(y_train_ner, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_test_ner = sequence.pad_sequences(y_test_ner, maxlen=MAX_LENGTH, truncating='post', padding='post')

print(X_train_features.shape)
X_train_features = sequence.pad_sequences(X_train_features, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_test_features = sequence.pad_sequences(X_test_features, maxlen=MAX_LENGTH, truncating='post', padding='post')
print(X_train_features.shape)

print(X_train_sents_bert.shape)
X_test_sents_bert = sequence.pad_sequences(X_test_sents_bert, maxlen=MAX_LENGTH, truncating='post', padding='post')
X_train_sents_bert = sequence.pad_sequences(X_train_sents_bert, maxlen=MAX_LENGTH, truncating='post', padding='post')
print(X_train_sents_bert.shape) 
# expand X_features dimension

X_train_features =  np.expand_dims(X_train_features, axis=2)
X_test_features =  np.expand_dims(X_test_features, axis=2)

# get the size of pos-tags, ner tags
TAG_VOCAB = len(list(idx2pos.keys()))
NER_VOCAB = len(list(idx2ner.keys()))


# reshape data for CRF
y_train_ner = y_train_ner[:, :, np.newaxis]
y_test_ner = y_test_ner[:, :, np.newaxis]


# ## pre-load the pretrained embeddings
# 
# as seen in previous studies such as Ma & Hovy 2016, loading the embedding layer with pretrained embedding vectors has been shown to improve network performance. here we initialize an embedding to zeros, and then load the embedding from the pretrained model (if it exists; it may not due to `Word2Vec` parameters).


# create embedding matrices from custom pretrained word2vec embeddings
word_embedding_matrix = np.zeros((MAX_VOCAB, WORDEMBED_SIZE))
c = 0
for word in word2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_vocab:
        c += 1
        # get the word vector
        word_vector = w2v_model[word]
        # slot it in at the proper index
        word_embedding_matrix[word2idx[word]] = word_vector
print("added", c, "vectors")



pos_embedding_matrix = np.zeros((TAG_VOCAB, POS_EMBED_SIZE))
c = 0
for word in pos2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if word in w2v_pvocab:
        c += 1
        # get the word vector
        word_vector = w2v_pmodel[word]
        # slot it in at the proper index
        pos_embedding_matrix[pos2idx[word]] = word_vector
print("added", c, "vectors")



# define model

# text layers : dense embedding > dropout > bi-LSTM
txt_input = Input(shape=(MAX_LENGTH,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, WORDEMBED_SIZE, input_length=MAX_LENGTH,
                      weights=[word_embedding_matrix],
                      name='txt_embedding', trainable=True, mask_zero=True)(txt_input)
txt_drpot = Dropout(DROPOUTRATE, name='txt_dropout')(txt_embed)

# pos layers : dense embedding > dropout > bi-LSTM
pos_input = Input(shape=(MAX_LENGTH,), name='pos_input')
pos_embed = Embedding(TAG_VOCAB, POS_EMBED_SIZE, input_length=MAX_LENGTH,
                      weights=[pos_embedding_matrix],
                      name='pos_embedding', trainable=True, mask_zero=True)(pos_input)
pos_drpot = Dropout(DROPOUTRATE, name='pos_dropout')(pos_embed)

# bert layer
bert_input = Input(shape=(MAX_LENGTH,768), name='bert_input')
bert_drpot = Dropout(DROPOUTRATE, name='bert_input')(bert_input)

# add auxiliary layer
auxiliary_input = Input(shape=(MAX_LENGTH,1), name='aux_input') #(None, 30, 1)

# merged layers : merge (concat, average...) word and pos > bi-LSTM > bi-LSTM
mrg_cncat = concatenate([txt_drpot, pos_drpot, bert_input], axis=2)
mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='mrg_bidirectional_1')(mrg_cncat)

# extra LSTM layer, if wanted
mrg_drpot = Dropout(DROPOUTRATE, name='mrg_dropout')(mrg_lstml)
mrg_lstml = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                          name='mrg_bidirectional_2')(mrg_lstml)

# merge BLSTM layers and extenal layer
mrg_cncat = concatenate([mrg_lstml, txt_drpot, auxiliary_input], axis=2)
# final linear chain CRF layer
crf = CRF(NER_VOCAB, sparse_target=True)
mrg_chain = crf(mrg_cncat)

model = Model(inputs=[txt_input, pos_input, bert_input, auxiliary_input], outputs=mrg_chain)

model.compile(optimizer='adam',
              loss=crf.loss_function,
              metrics=[crf.accuracy])



model.summary()



history = model.fit([X_train_sents, X_train_pos, X_train_sents_bert, X_train_features], y_train_ner,
                    batch_size=BATCH_SIZE,
                    epochs=MAX_EPOCHS,
                    verbose=2)

hist_dict = history.history


# save the model
# because we are using keras-contrib, we must save weights like this, and load into network
# (see decoding.ipynb)
save_load_utils.save_all_weights(model, '../model/crf_model.h5')
np.save('../model/hist_dict.npy', hist_dict)
print("models saved!\n")


preds = model.predict([X_test_sents, X_test_pos, X_test_sents_bert, X_test_features])



preds = np.argmax(preds, axis=-1)
preds.shape
print(preds[:5])

trues = np.squeeze(y_test_ner, axis=-1)
trues.shape

s_preds = [[idx2ner[t] for t in s] for s in preds]

s_trues = [[idx2ner[t] for t in s] for s in trues]


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
def bio_classification_report(y_true, y_pred):
    """
    from scrapinghub's python-crfsuite example
    
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O', 'PAD'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    
    print(y_pred_combined[:5])
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


print(bio_classification_report(s_trues, s_preds))




