import os
import pandas

a = open("Stocks.txt","r")

path = os.path.join("Data","^GSPC" + ".csv")
d = pandas.read_csv(path,header=0,index_col=0)
d.columns = ["^GSPC"]

for line in a:
	name = line.replace("\n","")
	
	path = os.path.join("Data",name + ".csv")
	
	try:
		data = pandas.read_csv(path,header=0,index_col=0)
		data.columns = [name]

		d = d.merge(data,how="inner",left_index=True,right_index=True)

	except:
		continue

outpath = os.path.join("Data.csv")

d.to_csv(outpath)
print d