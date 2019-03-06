#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing
# convert the original data to the required format
# output is 

from jnius import autoclass

inputf = open("../data/TrainData2.tsv", encoding = "ISO-8859-1")
data = inputf.readlines()[1:]
print(data[0])

class CMUPosTagger(object):
    """used to pos tag tweets"""

    model_filename = ('../resources/ark_tweet_nlp-20120919.model')
      
    def __init__(self):
        self._setup()
  
    def _setup(self):
        Model = autoclass('cmu.arktweetnlp.impl.Model')
        self._model = Model.loadModelFromText(self.model_filename)

        FeatureExtractor = autoclass('cmu.arktweetnlp.impl.features.FeatureExtractor')
        self._featureExtractor = FeatureExtractor(self._model, False)

        self._Sentence = autoclass('cmu.arktweetnlp.impl.Sentence')
        self._ModelSentence = autoclass('cmu.arktweetnlp.impl.ModelSentence')
        logger.debug('Loaded Twitter POS tagger using model <{}>.'.format(os.path.relpath(self.model_filename, _parentdir)))

        self._initialized = True


    def tagger(self, toks):
        if not getattr(self, '_initialized', False): self._setup()
        if not toks: return []

        sentence = self._Sentence()
        for tok in toks: sentence.tokens.add(tok)
        ms = self._ModelSentence(sentence.T())
        self._featureExtractor.computeFeatures(sentence, ms)
        self._model.greedyDecode(ms, False)

        tags = []
        for t in xrange(sentence.T()):
          tag = self._model.labelVocab.name(ms.labels[t])
          tags.append(tag) 
        return tags

tknzr = TweetTokenizer()
pos_tagger = CMUPosTagger()
for j, sentence in enumerate(data[:2]):
    li = sentence.strip().split('\t')
    tweet, begin, end, extraction = li[7], int(li[1]), int(li[2]), li[5]
    if tweet[begin:end] != extraction:
        raise Exception("extraction is not in range!")
    tweet_li = tknzr.tokenize(tweet)
    pos_li = CMUPosTagger.tagger(tweet_li)
    print(tweet_li)
    print(pos_li)
#   begin_idx = int(begin)
#   end_idx = int(end)
#   for i, token in enumerate(tweet_li):
#       if i in range(begin_idx, end_idx):
#           if i == begin_idx:
#               tag = "B-ADR"
#           else:
#               tag = "I-ADR"
#       else: tag = "O"
#       sentence_li.append([token, pos_li[i], tag])
#   data_li.append(sentence_li)

# def save_data(data_li, outputf):
#   for i, sentence_li in enumerate(data_li):
#       for j, token_li in enumerate(sentence_li):
#           if j == 0:
#               outputf.write("Sentence: {}".format(i)+","+",".join(token_li)+"\n")
#           else:
#               outputf.write(","+",".join(token_li)+"\n")
#   return 0




