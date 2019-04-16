"""
convert decoded format to submitted format
"""
from collections import defaultdict
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import sys


f_li = [
"../data/TrainData1.tsv",
"../data/TrainData2.tsv",
"../data/TrainData3.tsv",
"../data/TrainData4.tsv",
]

original_data = {}
for f in f_li:
	inputf = open(f, encoding = "utf-8")
	data = inputf.readlines()[1:]
	for line in data:
		line = line.strip()
		li = line.split("\t")
		tweet_li, tweet, extraction = li[0], li[6], li[4] ## tsv 1 and 2 and 3 and 4
		original_data[tweet] = tweet_li

data = defaultdict(lambda : defaultdict(list))
result_f = 'results/keras-biLSTM-CRF_sample2.csv'
with open(result_f) as inputf:
	i = 0
	for line in inputf:
		line = line.strip()
		if line.startswith("word"):
			data[i]["word"] = line.split(",")[1:]
		elif line.startswith("pred"):
			data[i]["pred"] = line.split(",")[1:]
		elif line.startswith("skip"):
			i += 1
		else:
			continue
print(data[1])

for k in data.keys():
	words = data[k]["word"]
	preds = data[k]["pred"]
	# print(original_data.keys())
	pred = data[k]["pred"]

	if len(words) != len(pred):
		print(words)
		w_li = []
		key = -10
		for i, w in enumerate(words):
			if w.startswith("\"") and words[i+1] == "\"":
				key = i
			if i == key:
				w_li.append(w[1:])
			elif i == key+1:
				continue
				# print(i, len(pred), len(words))
			else: w_li.append(w)
		words = w_li

	start, end = 0, 0
	spans = []
	for i in range(len(pred)):
		if  (i == 0 and pred[i]== "I-ADR") or (pred[i-1] == "O" and pred[i] == "I-ADR") or pred[i] == "B-ADR":
			start = i
		if ( (i+1 == (len(pred) -1) and "ADR" in pred[i]) or (i < (len(pred) - 1) and pred[i+1] == "O" and "ADR" in pred[i]) or 
			(i < (len(pred)-1) and i > 0 and "ADR" not in pred[i+1] and "ADR" not in pred[i-1] and "ADR" in pred[i]) ):
			end = i
			spans.append((start, end))
		if pred[i] == "O":
			continue
	data[k]["spans"] = spans 
	data[k]["text"] = [" ".join(words[start:end+1]) for start, end in spans]
	# print(data[k])
	# sys.exit()
	tweet = " ".join(list(filter(lambda x : x!= "", words)))
	if tweet not in original_data.keys():
		tweet = "\"{}\"".format(tweet)
		if tweet not in original_data.keys():
			# print(words)
			# print(tweet)
			# print("not in ")
			original_words = process.extractOne(tweet, original_data.keys())
			# print(original_words)
			tweet_0 = original_words[0]
			if len(tweet.split()) != len(tweet_0.split()):
				print(tweet)
				print(tweet_0) 
			tweet = tweet_0
	data[k]["tweet_id"] = original_data[tweet]
	data[k]["tweet"] = tweet

	s = ""
	char_spans = []
	words = tweet.split()
	i = 0
	while spans != []:
		if i == spans[0][0]:
			start = len(s)
		if i == spans[0][1]:
			s += words[i]
			end = len(s)
			char_spans.append((start, end))
			s += " "
			i += 1
			print(tweet)
			print(s)
			print(spans)
			print(char_spans)
			print(tweet[start:end+1])
			spans = spans[1:]
			continue
		s += words[i]
		s += " "
		i += 1

	data[k]["char_spans"] = char_spans

data_li = []
for k in data.keys():
	tweet = data[k]["tweet"]
	tweet_id = data[k]["tweet_id"]
	char_spans = data[k]["char_spans"]
	text = data[k]["text"]
	assert len(text) == len(char_spans)
	for i in range(len(char_spans)):
		data_li.append([tweet_id, tweet, char_spans[i][0], char_spans[i][1], text[i]])

with open("output.txt", "w") as outputf:
	for instance in data_li:
		outputf.write("\t".join(line))
		outputf.write("\n")










