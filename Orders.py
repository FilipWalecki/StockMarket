from ib_insync import *
import csv 
import pandas as pd
import pandas_datareader as web 
import datetime as time
import sqlite3
# util.startLoop()  # uncomment this line when in a notebook
        

class Trader:
        def __init__(self, conn, cursor):
                self.conn = conn
                self.cursor = cursor
                self.ib = IB()
                self.ib.connect('127.0.0.1', 7497, clientId=1)

        def buy_passed_stocks(self):
                stocks = []

                self.cursor.execute('''SELECT * FROM passed''')

                rows = self.cursor.fetchall()
                for row in rows:
                        stocks.append(str(row[0]))
                

                        
                for i in range (len(stocks)):

                        
                        contract = Stock(stocks[i] ,'SMART','USD')
                        print(contract)

                        order = MarketOrder('BUY', 10)

                        print(order)

                

                        trade = self.ib.placeOrder(contract , order)

                        print(trade)
                self.ib.run()
        def close_position(self):
                pass
        
                