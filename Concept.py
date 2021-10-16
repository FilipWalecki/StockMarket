from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Stock('AMD','SMART','USD')

order = LimitOrder('BUY', 5,112.12)

print(order)

ib.placeOrder(contract , order)

print(trade)