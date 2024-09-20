from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
import random

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




def load_fx(data_start=0,data_end=5000, window_size = 10, shift=2, pair='EURUSD'):
    # Load the dataset
    # df = pd.read_csv("dataset/EURUSD_Daily_200005300000_202405300000.csv", delimiter="\t")
    path = '/Users/krasimirtrifonov/Documents/GitHub/VisibilityGraph/data/forex/all'
    if pair == 'EURUSD':
        fn = 'EURUSD_Daily_200005300000_202405300000.csv'
    if pair == 'AUDUSD':
        fn = 'AUDUSD_Daily_200005300000_202405300000.csv'
    if pair == 'GBPUSD':
        fn = 'GBPUSD_Daily_200005300000_202405300000.csv'
    df = pd.read_csv(f'{path}/{fn}',delimiter="\t")
    # df = pd.read_csv(
    #     "/Users/krasimirtrifonov/Documents/GitHub/Transformer-Neural-Network/dataset/EURUSD_Daily_200005300000_202405300000.csv",
    #     delimiter="\t")

    # Extract the closing prices
    closing = df["<CLOSE>"].iloc[data_start:data_end]

    # Parameters
    # window_size = 10  # Example window size
    # test_size = 0.2   # Test set size

    scaler = StandardScaler()

    # X_train = scaler.fit_transform(X_train)

    # print(f"shift {shift}")
    # Create sliding window features and labels
    X, y, returns = [], [], []
    for i in range(len(closing) - window_size):
        window = closing[i:i + window_size].values
        target = 1 if closing[i + window_size] > closing[i + window_size - shift] else 0
        ret = closing[i + window_size] - closing[i + window_size - shift]
        # Standardize the data
        # features = scaler.fit_transform(window[:-1].reshape(-1, 1))
        features = scaler.fit_transform(closing[i:i + window_size - 1].diff().dropna().values.reshape(-1, 1))
        X.append(features)
        y.append(target)
        returns.append(ret)

    X = np.array(X).squeeze()
    y = np.array(y)
    returns = np.array(returns)

    return X, y, returns
# X, y = load_fx(data_start=0, data_end=5100, shift=1)