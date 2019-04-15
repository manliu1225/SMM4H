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
	print(original_data.keys())
	if " ".join(words) in original_data.keys():
		print(words)
		tweet = " ".join(words)
	else:
		original_words = process.extractOne(" ".join(words), original_data.keys())
		print(" ".join(words))
		print(original_words)
		tweet = original_words[0]
	data[i]["tweet_id"] = original_data[tweet]
	sys.exit()










