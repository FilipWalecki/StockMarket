from ib_insync import *
import csv 
import pandas as pd
import pandas_datareader as web 
import datetime as time
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

stocks = []
with open('currentorder.csv', 'r') as f:
            reader = csv.reader(f)
            lists = list(f)
            



for i in range (len(list)):

        
        contract = Stock(stocks[i] ,'SMART','GDP')
        print(contract)

        order = MarketOrder('SELL', 10)

        with open ('currentorder.csv','a',newline = '') as y:
                  writer = csv.writer(g)
                  writer.writerow(order)

        print(order)

        trade = ib.placeOrder(contract , order)

        print(trade)
ib.run()
