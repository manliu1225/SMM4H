#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing
# convert the original data to the required format
# output is 

from nltk.tokenize import TweetTokenizer
import os
import jnius_config
import logging
from collections import OrderedDict
_resources_dir = "./resources"
jnius_config.add_options('-Xmx512m', '-XX:ParallelGCThreads=2')
jnius_config.set_classpath(*(os.path.join(_resources_dir, jar) for jar in os.listdir(_resources_dir) if jar.endswith('.jar')))
os.environ['CLASSPATH'] = './resources/'

inputf = open("../data/TrainData2.tsv", encoding = "ISO-8859-1")
data = inputf.readlines()[1:]
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



tknzr = TweetTokenizer()
pos_tagger = CMUPosTagger()
data_dict = OrderedDict()
sentence_dict = OrderedDict()
for j, sentence in enumerate(data[:2]):
    li = sentence.strip().split('\t')
    tweet, begin, end, extraction = li[7], int(li[1]), int(li[2]), li[5]
    if tweet[begin:end] != extraction:
        raise Exception("extraction is not in the range!")
    # tweet_li = tknzr.tokenize(tweet)
    tweet_li = tweet.split()
    print(tweet_li)
    pos_li = pos_tagger.tagger(tweet_li)
    print(pos_li)

    ### process the tag
    tag_li = []
    t_t = ["_"] * len(tweet)
    for k in range(len(tweet)):
        if k < begin and tweet and tweet[k] != " ":
            t_t[k] = "O"
        if  begin <= k < end and tweet[k] != " ":
            t_t[k] = "ADR"
        if k >= end and tweet[k] != " ":
            t_t[k] = "O"

    t_t_li = ''.join(t_t).split("_")
    for k in range(len(tweet_li)):
        assert len(t_t_li) == len(tweet_li)
        if "ADR" in t_t_li[k]:
            tag_li.append("ADR")
        else: tag_li.append("O")
    print(tweet[begin:end])
    print(tag_li)
    sentence_dict["word"] = tweet_li
    sentence_dict["tag"] = tag_li
    sentence_dict["pos"] = pos_li
    data_dict[j] = sentence_dict
    sentence_dict = OrderedDict()
print(data_dict)

### convert data_dict to 3 columns
with open("../data/converted_file_TrainData2.csv", "w") as outputf:
    outputf.write("Sentence #,Word,POS,Tag\n")
    for idx, sentence_dict in data_dict.items():
        word_li = sentence_dict["word"] = tweet_li
        tag_li = sentence_dict["tag"]
        pos_li = sentence_dict["pos"]
        sentence_m = ["Sentence: {}".format(idx+1)] + [""]*(len(word_li)-1)
        for s in  zip(sentence_m, word_li, tag_li, pos_li):
            outputf.write(",".join(s)+"\n")






