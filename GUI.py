import pandas_datareader as web 
import pandas as pd
import datetime as time
from ib_insync import *

ib = IB()
print(time.date.today()-time.timedelta(days=2))
ib.connect('127.0.0.1', 7497, clientId=1)
test = web.DataReader('AAPL', 'yahoo','2022-01-27')['Close']
acc_vals = float([v.value for v in ib.accountValues() if v.tag == 'CashBalance' and v.currency == 'BASE'][0])
t = test.to_numpy()
print(t)
ammount =acc_vals*0.03/t[0]
print(round(ammount,0))

ib.run()



