import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas
import datetime
import numpy

def RP():
	file = os.path.join("Results","RPCalibration.csv")
	
	fig,ax = plt.subplots()

	plt.figure(1)
	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%d/%m/%Y"),dates)

	ax.plot(dates,numpy.cumsum(data["SnP Return"]),label="SP500")
	for i in range(0,45,10):
		ax.plot(dates,numpy.cumsum(data[str(i)]))

	#Formatting for plot
	#Format x-axis
	months = mdates.MonthLocator()
	ax.xaxis.set_major_locator(months)
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
	fig.autofmt_xdate()
	#Label Axis
	plt.xlabel("Month/Year")
	plt.ylabel("Cumulative Return")
	#Show legend
	plt.legend()


	plt.figure(2)
	file = os.path.join("Results","RPValidation.csv")
	
	fig,ax = plt.subplots()

	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%d/%m/%Y"),dates)

	ax.plot(dates,numpy.cumsum(data["SnP Return"]),label="SP500")
	for i in range(0,45,10):
		ax.plot(dates,numpy.cumsum(data[str(i)]))

	#Formatting for plot
	#Format x-axis
	months = mdates.MonthLocator()
	ax.xaxis.set_major_locator(months)
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
	fig.autofmt_xdate()
	#Label Axis
	plt.xlabel("Month/Year")
	plt.ylabel("Cumulative Return")
	#Show legend
	plt.legend()


	plt.show()



if __name__ == "__main__":
	RP()