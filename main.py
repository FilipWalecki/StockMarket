import analysis as a
import psutil
import GUI
import Orders
import os

#GAME PLAN
#RUN GUI-->analysis -- >second analysis ----.orders
if __name__ == "__main__":
    os.system("Trader Workstation")
    if 'tws.exe' in (i.name() for i in psutil.process_iter()) == True:
        run = a.analysis()
        run.runall()
        Orders.connection()
        Orders.buying()
    

    
    