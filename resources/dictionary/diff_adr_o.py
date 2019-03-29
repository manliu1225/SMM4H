f1 = 'namelist1.txt'
f2 = 'namelist2.txt'
f3 = 'disease.txt'

namelist1 = set([e.strip().lower() for e in open(f1).readlines()[1:] if e.strip() != ''])
namelist2 = set([e.strip().lower() for e in open(f2).readlines()[1:] if e.strip() != ''])
namelist_o = set([e.strip().lower() for e in open(f3).readlines()[1:] if e.strip() != ''])

namelist_adr = namelist1 | namelist2

print(len(namelist_adr))
print(len(namelist_o))

same = namelist_adr & namelist_o

clean_namelist_adr = namelist_adr - namelist_o
print(len(clean_namelist_adr))

with open('namelist_adr.txt', 'w') as inputf:
	inputf.write('@@@ADR\n')
	for line in clean_namelist_adr:
		inputf.write(line + '\n')
