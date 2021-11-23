from ast import Str
import numpy as np
from numpy.core.records import array
import pandas as pd
import pandas_datareader as web 
import datetime as time
import sqlite3
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.python.util.keras_deps import get_load_model_function


class Analysis:
   

    def __init__(self, conn, cursor):
        self.conn = conn
        self.cursor = cursor

        
        self.stocks =[]
        self.x_train =[]
        self.y_train =[]
        self.x_test = []
        self.predicted_prices = []
        self.actual_prices = []
        self.good_stocks = []

    def dataManipulation(self):
        #loading hte values from databse into a list
        # cursor = conn.cursor()


        self.cursor.execute('''SELECT * FROM stock''')

        rows = self.cursor.fetchall()
        for row in rows:
            self.stocks.append(str(row[0]))

        self.conn.commit()
        #testing if data for this stock exists
        

        #Loading the data
        
        for i in range(len(self.stocks)):
            try:
                self.x_test = []
                self.x_train = []
                self.y_train = []
                self.stock = self.stocks[i]
                
                start = time.datetime(2016,1,1)
                end  = time.datetime(2020,1,1)

                data = web.DataReader(self.stock , 'yahoo' , start,end)
                

                #Prepare Data
                scaler = MinMaxScaler(feature_range=(0,1))
                ScaledData = scaler.fit_transform(data['Close'].values.reshape(-1,1))

                PredictionDays = 80

               

                for x in range(PredictionDays,len(ScaledData)):
                    self.x_train.append(ScaledData[x-PredictionDays:x, 0])
                    self.y_train.append(ScaledData[x,0])

                self.x_train,self.y_train = np.array(self.x_train), np.array(self.y_train)
                self.x_train = np.reshape(self.x_train,(self.x_train.shape[0],self.x_train.shape[1],1))
                #build the model
                model = Sequential()

                model.add(LSTM(units =50,return_sequences=True,input_shape =(self.x_train.shape[1],1) ))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50 ,return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))

                model.compile(optimizer='adam', loss = 'mean_squared_error')
                model.fit(self.x_train,self.y_train ,epochs=25,batch_size =32)


                #Test the model accuracy

                test_start = time.datetime(2020,1,1)
                test_end = time.datetime.now()

                test_data = web.DataReader(self.stock, 'yahoo',test_start, test_end)
                self.actual_prices = test_data['Close'].values

                total_dataset = pd.concat((data['Close'],test_data['Close']),axis = 0)
                model_inputs = total_dataset[len(total_dataset)-len(test_data)-PredictionDays:].values
                model_inputs = model_inputs.reshape(-1,1)
                model_inputs = scaler.transform(model_inputs)

                #Prediction on test data


                for x in range(PredictionDays,len(model_inputs)):
                    self.x_test.append(model_inputs[x+1-PredictionDays:x,0])


                self.x_test = np.array(self.x_test)
                self.x_test = np.reshape(self.x_test,(self.x_test.shape[0],self.x_test.shape[1],1))

                self.predicted_prices = model.predict(self.x_test)
                self.predicted_prices = scaler.inverse_transform(self.predicted_prices)
                   
                #Predict Next Day
                
                real_data = [model_inputs[len(model_inputs)+1 - PredictionDays:len(model_inputs+1 ),0]]
                real_data = np.array(real_data)
                real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

                prediction = model.predict(real_data)
                prediction = scaler.inverse_transform(prediction)

                #predicitng the 4th day in order to see if the stock will rise or fall
                prediction2 = [model_inputs[len(model_inputs)+4 - PredictionDays:len(model_inputs+4 ),0]]
                prediction2 = np.array(prediction2)
                prediction2 = np.reshape(prediction2,(prediction2.shape[0],prediction2.shape[1],1))

                predicted2 = model.predict(prediction2)
                predicted2 = scaler.inverse_transform(predicted2 )

               
                
                print(f"Prediction:{prediction}")
                print(f"Prediction:{predicted2}")
                print(self.predicted_prices[-1])
                print(self.stock)
                
                '''plt.plot(self.actual_prices, color = 'red')
                plt.plot(self.predicted_prices,color ='green')
                plt.title(f'{self.stock} share price')
                plt.xlabel('time')
                plt.ylabel('Share price')
                plt.legend()
                plt.show()'''
                
                #print(f"Prediction:{self.stock}")
                
              
                
                #Checking if thew prediction was accurate
                if np.sum(self.predicted_prices)/np.sum(self.actual_prices) <= 1.01 and np.sum(self.predicted_prices)/np.sum(self.actual_prices) >= 0.97  and float(self.predicted_prices[-1])<float(predicted2) :
                     self.good_stocks.append(self.stock)
            except:
                pass
                
                
    def AddingToCsv(self): 
        #placing stocks to separate database
        
        for i in range(len(self.good_stocks)):
            self.cursor.execute('''INSERT INTO passed VALUES(?)''',(self.good_stocks[i],))
            self.conn.commit()
            
    def runall(self):
        self.dataManipulation()
        self.AddingToCsv()

#Ploting the predictions dont need it now might use it in the future

                
        



        

        

        
    
