import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas
import datetime
import numpy

def RP():
	#Plot Calibration Graphs
	file = os.path.join("SamplePredict.csv")
	
	fig,ax = plt.subplots()

	plt.figure(1)
	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),dates)

	ax.plot(dates,numpy.cumsum(data["^STI"]),label="STI")
	for i in range(0,50,1):
		try:
			ax.plot(dates,numpy.cumsum(data[str(i)]),label=str(i+10))
		except:
			continue

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
	plt.title("Replicating Portfolio(Calibration)")

	#Plot Validation Graphs
	plt.figure(2)
	file = os.path.join("ValidationPredict.csv")
	
	fig,ax = plt.subplots()

	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),dates)

	ax.plot(dates,numpy.cumsum(data["^STI"]),label="STI")
	for i in range(0,50,1):
		try:
			ax.plot(dates,numpy.cumsum(data[str(i)]),label=str(i+10))
		except:
			continue

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
	plt.title("Replicating Portfolio(Validation)")

	plt.show()

def IRP():
	#Plot Calibration Graphs
	file = os.path.join("SamplePredict2.csv")
	
	fig,ax = plt.subplots()

	plt.figure(1)
	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),dates)

	ax.plot(dates,numpy.cumsum(data["^GSPC"]),label="SP500")
	ax.plot(dates,numpy.cumsum(data["^^GSPC"]),label="SP500 Modified")
	for i in range(0,45,10):
		ax.plot(dates,numpy.cumsum(data[str(i)]),label=str(i+10))

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
	plt.title("Improved Replicating Portfolio(Calibration)")


	#Plot Validation Graphs
	plt.figure(2)
	file = os.path.join("ValidationPredict2.csv")

	fig,ax = plt.subplots()

	data = pandas.read_csv(file,header=0,index_col=0)

	dates = data.index
	dates = map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d"),dates)

	ax.plot(dates,numpy.cumsum(data["^GSPC"]),label="SP500")
	ax.plot(dates,numpy.cumsum(data["^^GSPC"]),label="SP500 Modified")
	for i in range(0,45,10):
		ax.plot(dates,numpy.cumsum(data[str(i)]),label=str(i+10))

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
	plt.title("Improved Replicating Portfolio(Validation)")


	plt.show()

def DeepFrontier():
	file = os.path.join("ValidationError.csv")
	
	fig,ax = plt.subplots()

	data = pandas.read_csv(file,header=0,index_col=0)

	y = data.index
	y = map(lambda y: y+10,y)

	ax.plot(data["Error"],y)

	#Formatting for plot
	#Label Axis
	plt.xlabel("Validation Error")
	plt.ylabel("Number of Stocks used")
	plt.gca().invert_yaxis()
	#Show legend
	plt.legend()
	plt.title("Deep Frontier")

	plt.show()

if __name__ == "__main__":
	RP()
	DeepFrontier()