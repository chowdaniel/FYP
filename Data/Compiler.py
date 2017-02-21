import os
import pandas

a = open("Stocks.txt","r")
d = pandas.DataFrame()

for line in a:
	name = line.replace("\n","")
	
	path = os.path.join(name + ".csv")
	
	try:
		data = pandas.read_csv(path,header=0,index_col=0)
		data.columns = [name]

		if len(data.index) == 504:
			d = d.merge(data,how="outer",left_index=True,right_index=True)
	except:
		continue

outpath = os.path.join("Merged.csv")

#d = d.fillna(method="backfill")
#d = d.fillna(method="ffill")

d.to_csv(outpath)
print d

#d1 = pandas.read_csv("Data/A.csv",header=0,index_col=0)
#d2 = pandas.read_csv("Data/AAL.csv",header=0,index_col=0)

#print d1
#print d2

#d1 = d1.merge(d2,how="outer",left_index=True,right_index=True)
#print d1
