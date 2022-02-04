from analysis import Analysis
import sqlite3
from Orders import Trader


#Main function
if __name__ == "__main__":
    # Initialize database connection
    conn = sqlite3.connect('stocks.db')
    cursor = conn.cursor()

    # Cleans up old results
    cursor.execute('''DELETE FROM passed''')
    conn.commit()
    
    # This part does the  analysis of the code
    analysisObj = Analysis(conn, cursor)
    analysisObj.runall()

    # this part is responisble for buinh and selling of the stocks
    traderObj = Trader(conn, cursor)
   
    traderObj.buy_passed_stocks()
    traderObj.close_position()
    
    

    # Close DB connection
    conn.close()
    

    
    