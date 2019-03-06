#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing
# convert the original data to the required format
# output is 

# from nltk.tokenize import TweetTokenizer

inputf = open("../data/TrainData2.tsv", encoding = "ISO-8859-1")
data = inputf.readlines()[1:]
print(data[0])

# tknzr = TweetTokenizer()

for j, sentence in enumerate(data[:2]):
	li = sentence.strip().split('\t')
	tweet, begin, end, extraction = li[7], int(li[1]), int(li[2]), li[5]
	if tweet[begin:end] != extraction:
		raise Exception("extraction is not in range!")
# 	tweet_li = tknzr.tokenize(tweet)
# 	pos_li = tweet_pos(tweet)
# 	begin_idx = int(begin)
# 	end_idx = int(end)
# 	for i, token in enumerate(tweet_li):
# 		if i in range(begin_idx, end_idx):
# 			if i == begin_idx:
# 				tag = "B-ADR"
# 			else:
# 				tag = "I-ADR"
# 		else: tag = "O"
# 		sentence_li.append([token, pos_li[i], tag])
# 	data_li.append(sentence_li)

# def save_data(data_li, outputf):
# 	for i, sentence_li in enumerate(data_li):
# 		for j, token_li in enumerate(sentence_li):
# 			if j == 0:
# 				outputf.write("Sentence: {}".format(i)+","+",".join(token_li)+"\n")
# 			else:
# 				outputf.write(","+",".join(token_li)+"\n")
# 	return 0




