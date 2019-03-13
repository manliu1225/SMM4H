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

inputf = open("../data/TrainData2-utf8.tsv", encoding = "utf-8")
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

miss = 0 # how many NEs are not in the begin ~ end
for j, sentence in enumerate(data):
    li = sentence.strip().split('\t')
    print(li)
    tweet, extraction = li[7], li[5]
    if extraction[0] == "\"" and extraction[-1] == "\"": extraction = extraction[1:-1]
    if extraction.lower() not in tweet.lower():
        # raise Exception("extraction is not in the range!")
        miss += 1
        continue
    begin = tweet.lower().index(extraction.lower())
    end = begin + len(extraction)
    if tweet[0] == "\"" and tweet[-1] == "\"":
        tweet = tweet[1:-1]
    # tweet_li = tknzr.tokenize(tweet)
    tweet_li = tweet.split()
    pos_li = pos_tagger.tagger(tweet_li)

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
    t_t_li = [x for x in t_t_li if x != ""]
    # print(t_t_li)
    # print(tweet_li)
    for k in range(len(tweet_li)):
        assert len(t_t_li) == len(tweet_li)
        if "ADR" in t_t_li[k]:
            tag_li.append("ADR")
        else: tag_li.append("O")
    sentence_dict["word"] = tweet_li
    sentence_dict["tag"] = tag_li
    sentence_dict["pos"] = pos_li
    data_dict[j] = sentence_dict
    sentence_dict = OrderedDict()

print("####there are {} missing ne".format(miss))
### convert data_dict to 3 columns
with open("../data/converted_file_TrainData2.csv", "w") as outputf:
    outputf.write("Sentence #,Word,POS,Tag\n")
    for idx, sentence_dict in data_dict.items():
        word_li = sentence_dict["word"]
        tag_li = sentence_dict["tag"]
        pos_li = sentence_dict["pos"]
        sentence_m = ["Sentence: {}".format(idx+1)] + [""]*(len(word_li)-1)
        for s in  zip(sentence_m, word_li, pos_li, tag_li):
            outputf.write(",".join(s)+"\n")
        outputf.write(",0,0,O\n")






