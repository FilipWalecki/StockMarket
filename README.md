# StockMarket

About:
As my NEA for A levels in Computer Science I decide to create an automated trading bot that would learn how to trade and then execute these trades all by its self.I've managed to sucessful complete this project by using multiple different libraries and the Interactive Brokers(IB) trading platfrom to execute all trades.The program has two versions that can be used.Both of these require the user to own an account with IB. One version allows to use the program without the need for real money.This can be used to play around with a accurate simulation of the stock market and learn about it. The second more risky version allows the user to trade with real money at their own risk. This is a link to a video that explains how the program works and shows it in action:
Important files:
-Main.py(Main part of the program)
-Orders.py(responsible for buying and selling stock )
-analysis.py(analises stock from the S&P500 list)
-Stocks.db(External database responsible for holding all the valuable information)
Libraries it uses:
-numpy
-pandas_datareader
-sklearn
-tensorflow
-matplotlib
-ib_insync
-sqlite3



