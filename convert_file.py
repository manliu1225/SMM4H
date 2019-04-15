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

inputf = open("../data/TrainData4.tsv", encoding = "utf-8")
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
    tweet, extraction = li[6], li[4] ## tsv 1 and 2 and 3 and 4
    # tweet, extraction = li[9], li[6] ## tsv 3
    # tweet, extraction = li[7], li[6] ## tsv 4
    # print(tweet)
    tweet_li = process(tweet)
    # sys.exit(0)
    tweet = " ".join(tweet_li)
    pos_li = pos_tagger.tagger(tweet_li)
    pos_n_li = nltk.pos_tag(tweet_li)

    if extraction == "": 
        begin = end = 100000
    else:
        extraction_li = process(extraction)
        extraction = " ".join(extraction_li)
        if extraction.lower() not in tweet.lower():
            print(extraction)
            print(tweet)
            raise Exception("extraction is not in the range!")
            # miss += 1
            # continue

        # extraction_li = tknzr.tokenize(extraction)
        extraction_li = extraction.split()
        extraction = " ".join(extraction_li)

        # get the begin and end
        begin = tweet.lower().index(extraction.lower())
        end = begin + len(extraction)

    ### process the tag
    tag_li = []
    t_t = ["_"] * len(tweet)
    for k in range(len(tweet)):
        if k < begin and tweet[k] != " ":
            t_t[k] = "O"
        if  begin <= k < end and tweet[k] != " ":
            t_t[k] = "ADR"
        if k >= end and tweet[k] != " ":
            t_t[k] = "O"

    t_t_li = ''.join(t_t).split("_")
    t_t_li = [x for x in t_t_li if x != ""]
    for k in range(len(tweet_li)):
        assert len(t_t_li) == len(tweet_li)
        if "ADR" in t_t_li[k]:
            tag_li.append("ADR")
        else: tag_li.append("O")
    sentence_dict["word"] = tweet_li
    sentence_dict["tag"] = tag_li
    sentence_dict["pos"] = pos_li
    sentence_dict["npos"] = [e[1] for e in pos_n_li]
    data_dict[j] = sentence_dict
    sentence_dict = OrderedDict()

### process duplicate ###
p_d = defaultdict(lambda : defaultdict(list))
for sentence_dict in data_dict.values():
    word_li = sentence_dict["word"]
    # print(word_li)
    p_d["\t".join(word_li)]["tag"].append(sentence_dict["tag"])
    p_d["\t".join(word_li)]["pos"] = sentence_dict["pos"]
    p_d["\t".join(word_li)]["npos"] = sentence_dict["npos"]

idx = 0
data_dict_new = defaultdict(lambda : defaultdict(list))
sentence_dict = defaultdict(list)
for k, v in p_d.items():
    t = v["tag"]
    p = v["pos"]
    np = v["npos"]
    tag_li = []
    if len(t) > 1: # there are more than 1 NEs in this tweet
        x = ""
        for i in range(len(t[0])):
            for j in range(len(t)):
                if t[j][i] != "O":
                    x = t[j][i]
                    break
                else: x = "O"
            tag_li.append(x)
    else: tag_li = t[0]
    new_tag_li = []
    for i, t in enumerate(tag_li):
        if i == 0 and t == "ADR":
            new_tag_li.append("B-ADR")
        elif t == "ADR" and tag_li[i-1] == "O":
            new_tag_li.append("B-ADR")        
        elif t == "ADR" and tag_li[i-1] == "ADR":
            new_tag_li.append("I-ADR")
        else:
            new_tag_li.append("O")
    data_dict_new[idx]["word"] = k.split("\t")
    data_dict_new[idx]["tag"] = new_tag_li
    data_dict_new[idx]["pos"] = p
    data_dict_new[idx]["npos"] = np
    # data_dict_new[idx] = sentence_dict
    idx += 1

# print(data_dict_new)

print("####there are {} missing ne".format(miss))
### convert data_dict to 3 columns
with open("./data/converted_file_TrainData4.csv", "w") as outputf:
    outputf.write("Sentence #\tWord\tPOS\tNPOS\tTag\t\n")
    for idx, sentence_dict in data_dict_new.items():
        word_li = sentence_dict["word"]
        tag_li = sentence_dict["tag"]
        pos_li = sentence_dict["pos"]
        npos_li = sentence_dict["npos"]
        sentence_m = ["Sentence: {}".format(idx+1)] + [""]*(len(word_li)-1)
        for s in  zip(sentence_m, word_li,  pos_li, npos_li, tag_li):
            outputf.write("\t".join(s)+"\n")
        outputf.write("\t0\t0\t0\tO\n")






