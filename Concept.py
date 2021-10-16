from ib_insync import *
import csv 
import pandas as pd
import pandas_datareader as web 
import datetime as time
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

stocks = []
with open('good_stocks.csv', 'r') as f:
            reader = csv.reader(f)
            lists = list(f)
            for i in range(len(lists)):
                stocks.append(lists[i].strip('\n'))



for i in range (len(stocks)):

        start = time.datetime(2021,10,15)
        end = time.datetime(2021,10,15)

        data = web.DataReader(stocks[i] , 'yahoo' , start)
        print(data)
        price = data['Close'].values
        print(price)
        contract = Stock(stocks[i] ,'SMART','USD')
        print(contract)

        order = LimitOrder('BUY', 1, round(price[1],2) )

        print(order)

        trade = ib.placeOrder(contract , order)

        print(trade)
ib.run()

        

        