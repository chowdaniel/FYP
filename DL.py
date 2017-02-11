import pandas_datareader.data as web
import datetime
import os

a = open("Stocks.txt","r")

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2017,1,1)

for line in a:
	name = line.replace("\n","")

	try:
		d = web.DataReader(name,"yahoo",start,end)
		path = os.path.join("Data",name + ".csv")

		d.to_csv(path,columns= ["Adj Close"])
	except:
		print name + " Error"