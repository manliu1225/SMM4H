f1 = "results/output.txt"

new_li = []
with open(f1) as inputf:
	for line in inputf:
		line = line.strip()
		tweet_id, tweet, start, end, cl, text = line.split("\t")
		new_li.append([tweet_id, start, end, cl, text])

with open("results/submit.txt", "w") as outputf:
	for line in new_li:
		outputf.write("\t".join(line))
		outputf.write("\n")


