import pandas
import numpy
import os

import statsmodels.api as sm

def Replicating(sample,validation,N):

	s_res = pandas.DataFrame(index=sample.index[1:])
	v_res = pandas.DataFrame(index=validation.index[1:])

	s_res["^GSPC"] = numpy.diff(numpy.log(sample.as_matrix(columns=["^GSPC"])),axis=0)
	v_res["^GSPC"] = numpy.diff(numpy.log(validation.as_matrix(columns=["^GSPC"])),axis=0)

	for n in N:
		#Import list of stocks sorted by increasing SSE
		symbols = open("Portfolio.csv","r")

		stocks = []
		for symbol in symbols:
			if symbol != "\n":
				stocks.append(symbol.replace("\n",""))
		symbols.close()
		
		#Select the stocks used for replicating portfolio
		chosen_stocks = []
		#10 Stocks with most communal info
		for i in range(n[0]):
			chosen_stocks.append(stocks[i])
		#n stocks with least communal info
		for i in range(n[1]):
			chosen_stocks.append(stocks[-i-1])
		chosen_stocks.append("^GSPC")


		#Calibration Phase====================================================

		#Import data for stocks
		data = pandas.DataFrame()

		for stock in chosen_stocks:
			prices = sample[stock]
			data[stock] = prices

		#Extract X and Y as numpy arrays
		X = data.as_matrix(columns=chosen_stocks[0:-1])
		Y = data.as_matrix(columns=chosen_stocks[-1:])

		#log prices
		X = numpy.log(X)
		Y = numpy.log(Y)

		#log returns
		X = numpy.diff(X,axis=0)
		Y = numpy.diff(Y,axis=0)

		X = sm.add_constant(X)

		model = sm.OLS(Y,X)
		results = model.fit()

		Y_pred = results.predict(X)
		s_res[n] = Y_pred

		#Validation Phase====================================================

		#Import data for stocks
		data = pandas.DataFrame()

		for stock in chosen_stocks:
			prices = validation[stock]
			data[stock] = prices

		#Extract X and Y as numpy arrays
		X = data.as_matrix(columns=chosen_stocks[0:-1])
		Y = data.as_matrix(columns=chosen_stocks[-1:])

		#log prices
		X = numpy.log(X)
		Y = numpy.log(Y)

		#log returns
		X = numpy.diff(X,axis=0)
		Y = numpy.diff(Y,axis=0)

		X = sm.add_constant(X)

		Y_pred = results.predict(X)
		v_res[n] = Y_pred

	s_res.to_csv("SamplePredict.csv")
	v_res.to_csv("ValidationPredict.csv")



if __name__ == "__main__":

	start1 = "2014-01-02"
	start2 = "2014-06-02"
	start3 = "2014-11-03"
	start4 = "2015-04-01"
	start5 = "2015-09-01"

	end1 = "2014-05-30"
	end2 = "2014-10-31"
	end3 = "2015-03-31"
	end4 = "2015-08-31"
	end5 = "2015-12-31"

	start6 = "2016-01-04"
	end6 = "2016-12-30"

	FILENAME = "Data.csv"
	imported_data = pandas.read_csv(FILENAME,header=0,index_col=0)

	sample = imported_data.loc[start1:end5]
	validation = imported_data.loc[start6:end6]

	Replicating(sample,validation,[(10,0),(10,10),(10,20),(10,30),(10,40)])