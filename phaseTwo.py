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