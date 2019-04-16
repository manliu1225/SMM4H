#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing
# convert the original data to the required format
# output is 

import nltk
import os
import sys
import jnius_config
import logging
import re
from collections import OrderedDict, defaultdict
_resources_dir = "./resources"
jnius_config.add_options('-Xmx512m', '-XX:ParallelGCThreads=2')
jnius_config.set_classpath(*(os.path.join(_resources_dir, jar) for jar in os.listdir(_resources_dir) if jar.endswith('.jar')))
os.environ['CLASSPATH'] = './resources/'

inputf = open("data/testDataST23_participants.txt", encoding = "utf-8")
data = inputf.readlines()
print(data[0])
logger = logging.getLogger(__name__)

from jnius import autoclass

class CMUPosTagger(object):
    """used to pos tag tweets"""

    model_filename = os.path.join("./resources", "ark_tweet_nlp-20120919.model")
    print(model_filename)

    def __init__(self):
        self._setup()
  
    def _setup(self):
        Model = autoclass("cmu.arktweetnlp.impl.Model")
        self._model = Model.loadModelFromText(self.model_filename)

        FeatureExtractor = autoclass('cmu.arktweetnlp.impl.features.FeatureExtractor')
        self._featureExtractor = FeatureExtractor(self._model, False)

        self._Sentence = autoclass('cmu.arktweetnlp.impl.Sentence')
        self._ModelSentence = autoclass('cmu.arktweetnlp.impl.ModelSentence')
        logger.debug('Loaded Twitter POS tagger using model <{}>.'.format(self.model_filename))

        self._initialized = True

    def __getstate__(self):
        d = dict(self.__dict__)

        del d['_Sentence']
        del d['_ModelSentence']
        del d['_featureExtractor']
        del d['_model']

        return d
  

    def __setstate__(self, d):
        self._setup()

        return True

    def tagger(self, tweet_li):
        if not getattr(self, '_initialized', False): self._setup()
        if not tweet_li: return []

        sentence = self._Sentence()
        for tok in tweet_li: sentence.tokens.add(tok)
        ms = self._ModelSentence(sentence.T())
        self._featureExtractor.computeFeatures(sentence, ms)
        self._model.greedyDecode(ms, False)

        tags = []
        for t in range(sentence.T()):
          tag = self._model.labelVocab.name(ms.labels[t])
          tags.append(tag) 
        return tags



# tknzr = TweetTokenizer()
pos_tagger = CMUPosTagger()
data_dict = OrderedDict()
sentence_dict = OrderedDict()

def process(tweet):
    '''
    process tweet and extraction'''
    if tweet[0] == "\"" and tweet[-1] == "\"":
        tweet = tweet[1:-1]
    tweet_li = tweet.split()
    # # tweet_li = tknzr.tokenize(tweet)
    # tweet_li = re.split(r'\s|([\,\?\!\:\"])', tweet)
    # tweet_li = list(filter(lambda x : x!=None, tweet_li))
    # tweet_li = list(filter(lambda x : x!="", tweet_li))
    # tweet_li_new = []
    # for i, s in enumerate(tweet_li):
    #     if len(s) > 1 and s[-1] == "." and not re.match(r"\.+",s):
    #         tweet_li_new.append(s[:-1])
    #         tweet_li_new.append('.')
    #     else: tweet_li_new.append(s)
    # # tweet_li = [re.sub(r"@[\d\w_]*", "@URL", x) for x in tweet_li]
    # tweet_li = tweet_li_new
    return tweet_li

miss = 0 # how many NEs are not in the begin ~ end
for j, sentence in enumerate(data):
    sentence = sentence.strip()
    # sentence = re.sub("amp;", "", sentence)
    li = sentence.strip().split('\t')
    tweet_id, tweet = li[0], li[1] ## tsv 1 and 2 and 3 and 4
    # tweet, extraction = li[9], li[6] ## tsv 3
    # tweet, extraction = li[7], li[6] ## tsv 4
    # print(tweet)
    tweet_li = process(tweet)
    # sys.exit(0)
    tweet = " ".join(tweet_li)
    pos_li = pos_tagger.tagger(tweet_li)
    pos_n_li = nltk.pos_tag(tweet_li)

    ### process the tag
    tag_li = ["O"] * len(tweet_li)
    sentence_dict["word"] = tweet_li
    sentence_dict["tag"] = tag_li
    sentence_dict["pos"] = pos_li
    sentence_dict["npos"] = [e[1] for e in pos_n_li]
    data_dict[j] = sentence_dict
    sentence_dict = OrderedDict()



print("####there are {} missing ne".format(miss))
### convert data_dict to 3 columns
with open("./data/converted_evluation.csv", "w") as outputf:
    outputf.write("Sentence #\tWord\tPOS\tNPOS\tTag\t\n")
    for idx, sentence_dict in data_dict.items():
        word_li = sentence_dict["word"]
        tag_li = sentence_dict["tag"]
        pos_li = sentence_dict["pos"]
        npos_li = sentence_dict["npos"]
        sentence_m = ["Sentence: {}".format(idx+1)] + [""]*(len(word_li)-1)
        for s in  zip(sentence_m, word_li,  pos_li, npos_li, tag_li):
            outputf.write("\t".join(s)+"\n")
        outputf.write("\t0\t0\t0\tO\n")






