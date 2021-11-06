from analysis import Analysis
import psutil
import sqlite3
from Orders import Trader
import os

#Runner
if __name__ == "__main__":
    # Initialize database connection
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()

    # Clean up old results
    # cursor.execute('''DELETE FROM passed''')
    # conn.commit()
    
    # Do analysis
    # analysisObj = Analysis(conn, cursor)
    # analysisObj.runall()

    # Buy now
    traderObj = Trader(conn, cursor)
    traderObj.buy_passed_stocks()

    # Close DB connection
    conn.close()
    

    
    