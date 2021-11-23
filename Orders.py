from ib_insync import *
import pandas as pd
import pandas_datareader as web 
import datetime as time
import sqlite3
from ib_insync.wrapper import Wrapper
# util.startLoop()  # uncomment this line when in a notebook
        

class Trader:
        def __init__(self, conn, cursor):
                self.conn = conn
                self.cursor = cursor
                self.ib = IB()
                self.ib.connect('127.0.0.1', 7497, clientId=1)
                self.stocks = []

        def buy_passed_stocks(self):
                

                self.cursor.execute('''SELECT * FROM passed''')
                

                rows = self.cursor.fetchall()
                for row in rows:
                        self.stocks.append(str(row[0]))
                


        
                print(self.stocks)
                        
                for i in range (len(self.stocks)):

                        
                        contract = Stock(self.stocks[i] ,'SMART','USD')
                        

                        order = MarketOrder('BUY', 10)
                        print(order)
                        

                

                        trade = self.ib.placeOrder(contract , order)
                        print(trade)


                        #placing the orders into a database
                
                      
                        self.cursor.execute('INSERT INTO Orders VALUES(?,?,?,?)',(self.stocks[i],'TRUE',time.date.today(),time.date.today()+time.timedelta(days=4)))
                        self.conn.commit()
                self.ib.run()
                        
                        

                
        def close_position(self):
                self.cursor.execute("SELECT ticker,Sell_date FROM Orders WHERE isBought = 'TRUE'")
                #aChange the true values to false
                rows = self.cursor.fetchall()
                
               
               
                for row in rows:
                       
                        if row[1] == str(time.date.today()):
                               
                                

                                contract = Stock(row[0],'SMART','USD')
                                order = MarketOrder('SELL',10)
                                trade = self.ib.placeOrder(contract,order)
                                self.cursor.execute('UPDATE Orders SET isBought = ? WHERE ticker = ? AND Sell_date <= ? ',('FALSE',row[0],time.date.today(),))
                                self.conn.commit()
                self.ib.run()
        def quickfix(self):
                
                self.cursor.execute("UPDATE Orders SET isBought = ?  WHERE Sell_date >= ?",('TRUE',time.date.today(),))
                self.conn.commit()

                                
        
                