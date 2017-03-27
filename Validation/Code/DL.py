import pandas_datareader.data as web
import datetime
import os

import pandas

#Model Parameters
FILENAME = "Merged.csv"

#Begin Code
imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

stocks = []

for stock in imported_data.columns:
	stocks.append(stock)
stocks.append("^GSPC")

start = datetime.datetime(2016,1,1)
end = datetime.datetime(2017,1,1)

for stock in stocks:

	try:
		d = web.DataReader(stock,"yahoo",start,end)
		path = os.path.join("Validation", stock + ".csv")

		d.to_csv(path,columns= ["Adj Close"])
	except:
		print name + " Error"

