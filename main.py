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

good_stocks =[]
#placing data from csv file to an array
stocks =[]

with open('file.csv', 'r') as f:
    reader = csv.reader(f)
    lists = list(f)
for i in range(len(lists)):
    stocks.append(lists[i].strip('\n'))



#Loading the data

for i in range(len(stocks)):
        company = stocks[i]
        start = time.datetime(2018,1,1)
        end  = time.datetime(2021,1,1)

        data = web.DataReader(company , 'yahoo' , start,end)
        

        #Prepare Data
        scaler = MinMaxScaler(feature_range=(0,1))
        ScaledData = scaler.fit_transform(data['Close'].values.reshape(-1,1))

        PredictionDays = 60

        x_train = []
        y_train = []

        for x in range(PredictionDays,len(ScaledData)):
            x_train.append(ScaledData[x-PredictionDays:x, 0])
            y_train.append(ScaledData[x,0])

        x_train,y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        #build the model
        model = Sequential()

        model.add(LSTM(units =50,return_sequences=True,input_shape =(x_train.shape[1],1) ))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50 ,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss = 'mean_squared_error')
        model.fit(x_train,y_train ,epochs=25,batch_size =32)


        #Test the model accuracy

        test_start = time.datetime(2021,1,1)
        test_end = time.datetime.now()

        test_data = web.DataReader(company, 'yahoo',test_start, test_end)
        actual_prices = test_data['Close'].values

        total_dataset = pd.concat((data['Close'],test_data['Close']),axis = 0)
        model_inputs = total_dataset[len(total_dataset)-len(test_data)-PredictionDays:].values
        model_inputs = model_inputs.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs)

        #Prediction on test data

        x_test = []

        for x in range(PredictionDays,len(model_inputs)):
            x_test.append(model_inputs[x+1-PredictionDays:x,0])


        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        #Ploting the predictions
        """
        plt.plot(actual_prices, color = 'red')
        plt.plot(predicted_prices,color ='green')
        plt.title(f'{company} share price')
        plt.xlabel('time')
        plt.ylabel('Share price')
        plt.legend()
        plt.show()"""
    
        #Predict Next Day
        real_data = [model_inputs[len(model_inputs)+1 - PredictionDays:len(model_inputs+1 ),0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        
        print(f"Prediction:{prediction}")
        #print(f"Real:{predicted_prices}")
        
                
        if np.sum(predicted_prices)/np.sum(actual_prices) <= 1.05 or np.sum(predicted_prices)/np.sum(actual_prices) >= 0.95:
            
            good_stocks.append(company)
            with open('good_stocks.csv','w', encoding='UTF8',newline ='') as g:

                writer = csv.writer(g)
                writer.writerow(company)
                
        
print(good_stocks)



        

        

        
