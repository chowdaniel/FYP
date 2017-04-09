import os
import pandas

a = open("Stocks.txt","r")
d = pandas.DataFrame()

for line in a:
	name = line.replace("\n","")
	
	path = os.path.join("Data",name + ".csv")
	
	try:
		data = pandas.read_csv(path,header=0,index_col=0)
		data.columns = [name]

		d = d.merge(data,how="outer",left_index=True,right_index=True)

		#if len(data.index) == 756:
			#d = d.merge(data,how="outer",left_index=True,right_index=True)
	except:
		continue

outpath = os.path.join("Data.csv")

#d = d.fillna(method="backfill")
#d = d.fillna(method="ffill")

d.to_csv(outpath)
print d