from ib_insync import *
import csv 
import pandas as pd
import pandas_datareader as web 
import datetime as time
import sqlite3
# util.startLoop()  # uncomment this line when in a notebook
def connection():
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)



def buying():

        stocks = []
        conn = sqlite3.connect('stocks.db')
        cursor = conn.cursor()

        cursor.execute('''SELECT * FROM passed''')

        rows =cursor.fetchall()
        for row in rows:
                stocks.append(str(row[0]))
        

                
        for i in range (len(stocks)):

                
                contract = Stock(stocks[i] ,'SMART','USD')
                print(contract)

                order = MarketOrder('BUY', 10)

                print(order)

        

                trade = ib.placeOrder(contract , order)

                print(trade)
        ib.run()

        

        