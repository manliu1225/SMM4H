"""
convert decoded format to submitted format
"""
from collections import defaultdict
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import sys


original_data = []
original_tweets = []

inputf = open("data/testDataST23_participants.txt", encoding = "utf-8")
data = inputf.readlines()
for line in data:
	line = line.strip()
	li = line.split("\t")
	tweet_li, tweet = li[0], li[1]
	original_data.append(tweet_li)
	original_tweets.append(tweet)

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
print(data[0])

for k in data.keys():
	words = data[k]["word"]
	preds = data[k]["pred"]
	# print(original_data.keys())
	pred = data[k]["pred"]

	### process "dept, "
	if len(words) != len(pred):
		# print(words)
		w_li = []
		key = -10
		for i in range(len(words)-1):
			if words[i].startswith("\"") and words[i+1] == "\"":
				key = i
			if i == key:
				w_li.append(words[i][1:])
			elif i == key+1:
				continue
				# print(i, len(pred), len(words))
			else: w_li.append(words[i])
		words = w_li

	### get start and end of tokens, start may equal end
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
	data[k]["tweet_id"] = original_data[k]
	tweet = original_tweets[k]
	data[k]["tweet"] = tweet

	### use fuzzy to extract the original tweet
	# tweet = " ".join(list(filter(lambda x : x!= "", words)))
	# if tweet not in original_data.keys():
	# 	tweet = "\"{}\"".format(tweet)
	# 	if tweet not in original_data.keys():
	# 		# print(words)
	# 		# print(tweet)
	# 		# print("not in ")
	# 		original_words = process.extractOne(tweet, original_data.keys())
	# 		# print(original_words)
	# 		tweet_0 = original_words[0]
	# 		if len(tweet.split()) != len(tweet_0.split()):
	# 			print(tweet)
	# 			print(tweet_0) 
	# 		tweet = tweet_0
	# data[k]["tweet_id"] = original_data[tweet]
	# data[k]["tweet"] = tweet

	### extract ADR begin and end, end is the end+1 of list
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
			spans = spans[1:]
			continue
		s += words[i]
		s += " "
		i += 1

	data[k]["class"] = "ADR"

	### If there are no ADR in tweet
	if char_spans == []: 
		char_spans = [("-", "-")]
		data[k]["class"] = "noADR"
	if data[k]["text"] == []:data[k]["text"] = ["-"]
	data[k]["char_spans"] = char_spans

# print(data)
data_li = []
for k in data.keys():
	tweet = data[k]["tweet"]
	tweet_id = data[k]["tweet_id"]
	char_spans = data[k]["char_spans"]
	text = data[k]["text"]
	cl = data[k]["class"]
	assert len(text) == len(char_spans)
	for i in range(len(char_spans)):
		start = char_spans[i][0]
		end = char_spans[i][1]
		### process text endswith .
		if text[i].endswith("."):
			text[i] = text[i][:-1]
			end = end -1 
		data_li.append([tweet_id, tweet, str(start), str(end), cl, text[i]])

with open("results/output.txt", "w") as outputf:
	for instance in data_li:
		print(instance)
		outputf.write("\t".join(instance))
		outputf.write("\n")










