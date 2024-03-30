# 1. Import the libraries

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
from keras.models import model_from_json

from pandas_datareader import data as pdr

import yfinance as yf

yf.pdr_override()  # <== that's all it takes :-)
import datetime as dt
# import Scaler here
from datetime import date


# comp_name = comp_name
def predict_stock(date_, comp_name):
    try:
        print(date_, comp_name)
        df = yf.download(comp_name, start="2020-01-01", end="2023-07-20")
        print(df)
        print(df.shape)
        # df = pdr.get_data_yahoo(comp_name, start="2020-01-01", end="2023-07-20")
        data = df.filter(['Close'])
        dataset = data.values
        format_str = '%d%m%Y'  # The format
        datetime_obj = dt.datetime.strptime(date_, format_str)
        end_date = datetime_obj.date()
        print(end_date)
        # import model
        Model_name = comp_name + '.h5'
        json = comp_name + '.json'
        json_file = open(json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(Model_name)
        # Get the quote
        apple_quote = pdr.get_data_yahoo(comp_name, start="2020-05-01", end=end_date)
        # apple_quote = web.DataReader(comp_name, data_source='yahoo', start='2012-01-01', end=end_date)
        # Create a new dataframe
        new_df = apple_quote.filter(['Close'])
        # Get teh last 60 day closing price
        last_60_days = new_df[-60:].values
        # Scale the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(dataset)

        last_60_days_scaled = scaler.transform(last_60_days)
        # Create an empty list
        X_test = []
        # Append teh past 60 days
        X_test.append(last_60_days_scaled)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Getting the models predicted price values
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)  # Undo scaling
        print("Predicted Stock Value:")
        print(predictions)

        from datetime import timedelta
        # print(strend_date)
        ss = (end_date - timedelta(1)).strftime('%Y-%m-%d')
        # Get the quote
        # Convert the given date string to a datetime object
        from datetime import datetime
        given_date = datetime.strptime(str(end_date), '%Y-%m-%d').date()
        # Get the current date
        current_date = datetime.now().date()
        print(given_date, current_date)
        if given_date > current_date:
            return predictions, "no"
        elif given_date == current_date:
            apple_quote2 = pdr.get_data_yahoo(comp_name, start=ss, end=end_date)
            return predictions, apple_quote2['Close']
        else:
            apple_quote2 = pdr.get_data_yahoo(comp_name, start=ss, end=end_date)
            return predictions, apple_quote2['Close']
    except Exception as e:
        print(e)
        return predictions, "no"
