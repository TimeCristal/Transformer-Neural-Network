from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import random
from scipy.linalg import hankel

def set_seed(seed):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fx(data_start=0, data_end=5000, window_size=10, shift=1, pair='EURUSD'):
    # Load the dataset
    # df = pd.read_csv("dataset/EURUSD_Daily_200005300000_202405300000.csv", delimiter="\t")
    path = '/Users/krasimirtrifonov/Documents/GitHub/VisibilityGraph/data/forex/all'
    if pair == 'EURUSD':
        fn = 'EURUSD_Daily_200005300000_202405300000.csv'
    if pair == 'AUDUSD':
        fn = 'AUDUSD_Daily_200005300000_202405300000.csv'
    if pair == 'GBPUSD':
        fn = 'GBPUSD_Daily_200005300000_202405300000.csv'
    if pair == 'USDJPY':
        fn = 'USDJPY_Daily_200005300000_202405300000.csv'
    if pair == 'USDCHF':
            fn = 'USDCHF_Daily_200005300000_202405300000.csv'
    if pair == 'USDCAD':
        fn = 'USDCAD_Daily_200005300000_202405300000.csv'



    df = pd.read_csv(f'{path}/{fn}', delimiter="\t")
    # df = pd.read_csv(
    #     "/Users/krasimirtrifonov/Documents/GitHub/Transformer-Neural-Network/dataset/EURUSD_Daily_200005300000_202405300000.csv",
    #     delimiter="\t")

    # Extract the closing prices
    low = df["<LOW>"].iloc[data_start:data_end]
    high = df["<HIGH>"].iloc[data_start:data_end]
    closing = df["<CLOSE>"].iloc[data_start:data_end]

    # Parameters
    # window_size = 10  # Example window size
    # test_size = 0.2   # Test set size

    scaler = StandardScaler()

    # X_train = scaler.fit_transform(X_train)

    # print(f"shift {shift}")
    # Create sliding window features and labels
    X, y_high, y_low, y_close, returns = [], [], [], [], []
    for i in range(len(closing) - window_size):
        window = closing[i:i + window_size].values
        target_high = 1 if high[i + window_size] > high[i + window_size - shift] else 0
        target_low = 1 if low[i + window_size] > low[i + window_size - shift] else 0
        target_close = 1 if closing[i + window_size] > closing[i + window_size - shift] else 0
        return_shift = 1  # return is always tomorrow minus today
        # ret = closing[i + window_size] - closing[i + window_size - return_shift]
        # percent change
        ret = ((closing[i + window_size] - closing[i + window_size - return_shift])/closing[i + window_size - return_shift])*100
        # Standardize the data
        # features = scaler.fit_transform(window[:-1].reshape(-1, 1))
        features = scaler.fit_transform(closing[i:i + window_size - 1].diff().dropna().values.reshape(-1, 1))
        X.append(features)

        y_high.append(target_high)
        y_low.append(target_low)
        y_close.append(target_close)

        returns.append(ret)

    X = np.array(X).squeeze()
    y_high = np.array(y_high)
    y_low = np.array(y_low)
    y_close = np.array(y_close)
    returns = np.array(returns)

    return X, y_high, y_low, y_close, returns
# X, y = load_fx(data_start=0, data_end=5100, shift=1)

def data_atom(array, window_size, shift):
    # Very simple and efficient way to prepare sliding window data with No loops
    # window_size = 18
    # shift = 3
    # orderList = 1:100;
    # array = np.random.rand(100)
    # orderList = np.arange(100)+1
    # print(f'np.shape(orderList) : {np.shape(array)}')
    # 1. Create Sliding window
    m = hankel(array[0:window_size], array[window_size - 1:]).T
    # print(f'm  : {np.shape(m)}')

    # 2. Create Labels form last Column and shifted
    y = (m[shift:, -1] > m[:-shift, -1]).astype(int)
    # print(f'y : {np.shape(y)}')

    # 3. Calculate returns (difference between consecutive rows of the last column)
    returns = m[1:, -1] - m[:-1, -1]
    # Adjust the length of returns to match the length of y
    returns = returns[-len(y):]
    # print(f'returns : {np.shape(returns)}')

    # 4. Create x: remove 'shift' number of rows from the bottom and the last column
    x = m[:-shift, :-1]
    # print(f'x : {np.shape(x)}')

    # 5. Make it stationary (difference between consecutive columns)
    x_dif = x[:, 1:] - x[:, :-1]
    # print(f'x_dif : {np.shape(x_dif)}')

    # 6. By Rows Standartization
    x_dif_std = (x_dif - x_dif.mean(axis=1, keepdims=True)) / x_dif.std(axis=1, keepdims=True)
    # print(f'x_dif_std : {np.shape(x_dif_std)}')
    return x_dif_std, y, returns


# Assuming X_Open, X_High, X_Low, X_Close are of shape (64, N)
def prepare_3d_matrix(X_Open, X_High, X_Low, X_Close):
    # Reshape each to (8, 8, N)
    X_Open_reshaped = X_Open.reshape(8, 8, -1)
    X_High_reshaped = X_High.reshape(8, 8, -1)
    X_Low_reshaped = X_Low.reshape(8, 8, -1)
    X_Close_reshaped = X_Close.reshape(8, 8, -1)

    # Stack along a new axis (axis=0) to create a 4*8*8*N matrix
    X_3d = np.stack([X_Open_reshaped, X_High_reshaped, X_Low_reshaped, X_Close_reshaped], axis=0)

    return X_3d  # Shape will be (4, 8, 8, N)

def load_fx_3d(window_size=66, shift=3, pair = 'EURUSD'):
    path = '/Users/krasimirtrifonov/Documents/GitHub/VisibilityGraph/data/forex/all'
    if pair == 'EURUSD':
        fn = 'EURUSD_Daily_200005300000_202405300000.csv'
    if pair == 'AUDUSD':
        fn = 'AUDUSD_Daily_200005300000_202405300000.csv'
    if pair == 'GBPUSD':
        fn = 'GBPUSD_Daily_200005300000_202405300000.csv'
    if pair == 'USDJPY':
        fn = 'USDJPY_Daily_200005300000_202405300000.csv'
    if pair == 'USDCHF':
            fn = 'USDCHF_Daily_200005300000_202405300000.csv'
    if pair == 'USDCAD':
        fn = 'USDCAD_Daily_200005300000_202405300000.csv'

    df = pd.read_csv(f'{path}/{fn}', delimiter="\t")

    close_price = df['<CLOSE>'].values
    open_price = df['<OPEN>'].values
    high_price = df['<HIGH>'].values
    low_price = df['<LOW>'].values

    X_Close, y, returns = data_atom(array=close_price, window_size=window_size, shift=shift)
    X_Open, _, _ = data_atom(array=open_price, window_size=window_size, shift=shift)
    X_High, _, _ = data_atom(array=high_price, window_size=window_size, shift=shift)
    X_Low, _, _ = data_atom(array=low_price, window_size=window_size, shift=shift)

    X_3d_matrix = prepare_3d_matrix(X_Open, X_High, X_Low, X_Close)

    return X_3d_matrix, y, returns