from ib_insync import *
import datetime as time
import pandas_datareader as web 

        

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
        
                
                for i in range (len(self.stocks)) :
                        #Adjusting the number of stocks bought based on account value
                        price = web.DataReader(self.stocks[i], 'yahoo',time.date.today()-time.timedelta(days=2))['Close']
                        acc_vals = float([v.value for v in self.ib.accountValues() if v.tag == 'CashBalance' and v.currency == 'BASE'][0])
                        converting_to_list = price.to_numpy()
                        ammount =acc_vals*0.03/converting_to_list[0]
                        final = round(ammount,0)
                        print(final)
                        
                        
                        contract = Stock(self.stocks[i] ,'SMART','USD' )

                        order = MarketOrder('BUY', final)
                        print(order)
        
                        trade = self.ib.placeOrder(contract , order)
                        
                        print(trade)
                        #placing the orders into a database
                        self.cursor.execute('INSERT INTO Orders VALUES(?,?,?,?,?)',(self.stocks[i],'TRUE',time.date.today(),time.date.today()+time.timedelta(days=4),final))
                        self.conn.commit()
        
                
                self.ib.run()
                      
                        
                      
        def close_position(self):
                self.cursor.execute("SELECT ticker,Sell_date,AmmountBought FROM Orders WHERE isBought = 'TRUE'")
                #aChange the true values to false
                rows = self.cursor.fetchall()
                
               
               
                for row in rows:
                       
                        if row[1] <= str(time.date.today()):
                               
                        

                                contract = Stock(row[0],'SMART','USD')
                                order = MarketOrder('SELL',row[2])
                                trade = self.ib.placeOrder(contract,order)
                                self.cursor.execute('UPDATE Orders SET isBought = ? WHERE ticker = ? AND Sell_date <= ? ',('FALSE',row[0],time.date.today(),))
                                self.conn.commit()
                                
                                print(trade)
                
                self.ib.run()

                                
        
                
