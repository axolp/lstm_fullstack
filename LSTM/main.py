import pandas as pd
import numpy as np
from lstm_from_scratch import LSTM


def predict(path, current_price, previous_price):

    lstm = LSTM(1, 1, 0.05, 0.7, "model.json")
    h, c, (f_gate, i_gate, c_gate, o_gate, c) = lstm.forward_propagation(current_price, previous_price, 0)
    return h


    

file_path = 'btc_hourly copy.csv'

# Wczytaj dane jako stringi
df = pd.read_csv(file_path, dtype=str)

# Usuń przecinki z kolumny 'High' i przekonwertuj na float
df['High'] = df['High'].str.replace(',', '').astype(float)

# Przekształć kolumnę 'High' na tablicę NumPy
data= np.array(df['High'])
mean = np.mean(data)
std = np.std(data)

lstm = LSTM(1, 1, 0.05, 0.9)

lstm.train(
    data, sequence_length= 14, epochs= 20, lr= 0.015, n= 30,
    optymalization_method= "batch gradient descent"
    )


lstm.predict(63000, lstm.h_prev, lstm.c_prev)





