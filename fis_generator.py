from math import e
import numpy as np
import random

#num_prod = 10
#maxfq = 30
dec_fact = 4 #decreasing factor
min_fract = 10
max_fract = 5


def writeFisFile(num_prod,maxfq,path):
	fq_disp = np.zeros(num_prod)
	file = open(path,"w")
	file.write("Num. prods = "+str(num_prod)+"\n")
	for i in range(0,num_prod):
		freq = int(maxfq*e**(-i/(num_prod/dec_fact)))
		file.write("{ "+str(i)+" }\t\t"+str(freq)+"\n")
		try:
			fq_disp[i] = freq - random.randrange(int(freq/min_fract),int(freq/max_fract)) #numero massimo di volte in cui e` stato venduto con altri
		except:
			fq_disp[i] = 0

	for i in range(0,num_prod):
		low = int(fq_disp[i]/6)
		high = fq_disp[i]  # low,high : range di selezione del numero di altri prodotti con cui e` stato acquistato
		if low < high:
			times = random.randrange(low,high)
		else:
			continue
		if i + times >= num_prod:
			remaining =  num_prod - i
			times = random.randrange(0,remaining)
		if times == 0:
			continue
		previous = np.full(times+1,i) # array che serve per far si che non vengano riscelti gli stessi valori
		for k in range(0,times):
			j = random.randrange(i+1,num_prod)
			while j in previous:
				j = random.randrange(i+1,num_prod)
			previous[k+1] = j
			minor = min(fq_disp[i],fq_disp[j])
			if minor > 1:
				freq = random.randrange(1,minor)
				fq_disp[i]-=freq
				fq_disp[j]-=freq
				file.write("{ "+str(i)+" , "+str(j)+" }\t\t"+str(freq)+"\n")

	file.close()


def readFisFile(path):
	file = open(path,"r")
	line = file.readline()
	num_prod = [int(s) for s in line.split() if s.isdigit()][0]
	fis = np.zeros((num_prod,num_prod),int) # 1000 max product
	line = file.readline()
	k = [int(s) for s in line.split() if s.isdigit()]
	while len(k) == 2:
		prod = k[0]
		freq = k[1]
		fis[prod,prod] = freq
		line = file.readline()
		k = [int(s) for s in line.split() if s.isdigit()]

	while line:
		p1 = k[0]
		p2 = k[1]
		freq = k[2]
		fis[p1,p2] = freq
		fis[p2,p1] = freq
		fis[p1,p1] -= freq
		fis[p2,p2] -= freq
		line = file.readline()
		k = [int(s) for s in line.split() if s.isdigit()]

	file.close()
	return fis




