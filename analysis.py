from ast import Str
from os import lseek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web 
import datetime as time
import csv



from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM
from tensorflow.python.util.keras_deps import get_load_model_function
from threading import Thread


class analysis():
    def __init__(self):
        self.stock = ""
        self.stocks =[]
        self.x_train =[]
        self.y_train =[]
        self.x_test = []
        self.predicted_prices = []
        self.actual_prices = []
        self.good_stocks = []
        
    def dataManipulation(self):
        #placing data from csv file to an array
        with open('file.csv', 'r') as f:
            reader = csv.reader(f)
            lists = list(f)
        for i in range(len(lists)):
            self.stocks.append(lists[i].strip('\n'))



        #Loading the data

        for i in range(len(self.stocks)):
                self.x_test = []
                self.x_train = []
                self.y_train = []
                self.stock = self.stocks[i]
                start = time.datetime(2018,1,1)
                end  = time.datetime(2021,1,1)

                data = web.DataReader(self.stock , 'yahoo' , start,end)
                

                #Prepare Data
                scaler = MinMaxScaler(feature_range=(0,1))
                ScaledData = scaler.fit_transform(data['Close'].values.reshape(-1,1))

                PredictionDays = 60

               

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

                test_start = time.datetime(2021,1,1)
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

                predicted_prices = model.predict(self.x_test)
                predicted_prices = scaler.inverse_transform(predicted_prices)

            
            
                #Predict Next Day
                real_data = [model_inputs[len(model_inputs)+1 - PredictionDays:len(model_inputs+1 ),0]]
                real_data = np.array(real_data)
                real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

                prediction = model.predict(real_data)
                prediction = scaler.inverse_transform(prediction)
                
                print(f"Prediction:{prediction}")
                print(f"Prediction:{self.stock}")
                #print(f"Real:{predicted_prices}")
                if np.sum(self.predicted_prices)/np.sum(self.actual_prices) <= 1.01 and np.sum(self.predicted_prices)/np.sum(self.actual_prices) >= 0.99:
                     self.good_stocks.append[self.stock]
                
    def AddingToCsv(self):      
        for i in range(len(self.good_stocks)):
            with open('good_stocks.csv','a', newline="") as g:

                writer = csv.writer(g)
                writer.writerow([self.good_stocks[i]])
    def runall(self):
        Thread(target= self.AddingToCsv()).start()
        Thread(target= self.dataManipulation()).start()


test = analysis()
test.runall()


#Ploting the predictions
"""
                plt.plot(actual_prices, color = 'red')
                plt.plot(predicted_prices,color ='green')
                plt.title(f'{company} share price')
                plt.xlabel('time')
                plt.ylabel('Share price')
                plt.legend()
                plt.show()"""

                
        



        

        

        
