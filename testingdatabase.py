import sqlite3
import csv
con = sqlite3.connect('stocks.db')
cur = con.cursor()
stocks =[]
# Create table

with open('file.csv', 'r') as f:
            reader = csv.reader(f)
            lists = list(f)
            for i in range(len(lists)):
                stocks.append(lists[i].strip('\n'))

# Insert a row of data
for i in range (len(stocks)):
    cur.execute("INSERT INTO stock VALUES (?)",
    (stocks[i],)) 

# Save (commit) the changes
con.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
con.close()

