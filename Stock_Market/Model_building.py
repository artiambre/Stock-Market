# 1. Import the libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')
# 2. Get the stock quote

end_date = '2023-07-01'
# end_date =  date.today()
start_date = '2020-12-17'
from pandas_datareader import data as pdr

import yfinance as yf

yf.pdr_override()  # <== that's all it takes :-)


def model_building(comp_name,market):
    if market == "bse":
        comp_name =comp_name.capitalize()+".BS"
    elif market == "nse":
        comp_name = comp_name.capitalize() + ".NS"
    try:
        df = yf.download(comp_name, start=start_date, end=end_date)
        print(df)
        print(df.shape)

        # Visualize the closing price history
        plt.figure(figsize=(16, 8))
        plt.title('Close Price History of' + comp_name)
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.switch_backend('agg')
        plt.show()

        # Create a new dataframe with only the 'Close' column
        data = df.filter(['Close'])
        # Converting the dataframe to a numpy array
        dataset = data.values
        # Get /Compute the number of rows to train the model on
        training_data_len = math.ceil(len(dataset) * .8)

        # Scale the all of the data to be values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(dataset)
        scaled_data = scaler.transform(dataset)

        # 3. Create the scaled training data set
        train_data = scaled_data[0:training_data_len, :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # 4. Build the LSTM network model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # 5. Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 6. Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # 7. Test data set
        test_data = scaled_data[training_data_len - 60:, :]
        # 8. Create the x_test and y_test data sets
        x_test = []
        y_test = dataset[training_data_len:,
                 :]  # Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])

        # Convert x_test to a numpy array
        x_test = np.array(x_test)

        # Reshape the data into the shape accepted by the LSTM
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Getting the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)  # Undo scaling

        ### serialize model to JSON
        model_json = model.to_json()
        json = comp_name + '.json'
        with open(json, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        Model_name = comp_name + '.h5'
        model.save_weights(Model_name)
        print("Saved model to disk")

        # Calculate/Get the value of RMSE
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

        # Plot/Create the data for the graph
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        return "Model trained and build successfully.."
    except Exception as e:
        print(e)
        return "Entered script not available."

# a=input(print("Enter Company Name"))
# b=input(print("Stock Marmekt"))
# model_building(a,b)