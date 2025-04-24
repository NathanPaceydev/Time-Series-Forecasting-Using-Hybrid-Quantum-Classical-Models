import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model_optimizer import train_and_evaluate
import itertools

# This script downloads stock data, processes it, and trains a model to predict future prices.

experiment_notes = "Quantum sim: " # Notes for the experiment

#-----------HYPER ParamETERS------------------
# Use a window size (e.g., 20)
WINDOW_SIZE = 20

# ---- quantum circuit parameters ----------
"""
n_qubits = 2 # number of qubits
q_depth  = 2 # number of layers
n_rot_params = 3  # <--- number of rotation parameters per qubit (e.g. 1 for RY, 3 for Rot)
"""

## iterate over these parameters ##
# Define hyperparameter ranges to iterate over
n_qubit_options = [2, 3, 4, 5, 6]
q_depth_options = [1, 2, 3, 4, 5, 6, 7]
n_rot_params_options = [1, 2, 3]

# wether to use quantum or classical model
use_quantum = True

#---------- classical model parameters ----------


# === Model Config ===
# Set model hyperparameters
input_size = 1
hidden_size = 16
output_size = 1

# draw the QML or ML model graph

learning_rate = 0.001

num_epochs = 10
use_dropout=False
dropout_rate= 0.0 # 0.2 to 0.5 is common for LSTM/GRU
use_layernorm=False 

# === Early Stopping Config ===
early_stop_patience = 2 # Number of epochs to wait for improvement
epochs_no_improve = 0

# Type of ML unit to use
ML_unit_type = "LSTM"  # "LSTM", "GRU", or "RNN" can havy many layers
num_ML_layers = 4  # Number of LSTM/GRU/RNN layers

post_quantum_activation = None  # "ReLU", "Tanh", "Sigmoid", or None
skip_connection = "concat"  # "concat" or "add"
output_activation = None # "ReLU", "Tanh", "Sigmoid", "Softmax", or None

#--------------------------------------------------



# Set the random seed for reproducibility
# This is important for consistent results across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make operations deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)



# ✅ List of tickers you want to load
tickers = ["AAPL", "MSFT", "GOOGL"]

# ✅ Dictionary to hold processed DataFrames per ticker
ticker_data = {}

for ticker in tickers:
    print(f"📥 Downloading data for {ticker}...")
    
    df = yf.download(ticker, start="2015-01-01", end="2024-12-31", interval="1d")

    # Flatten MultiIndex if necessary (e.g., from yf multi-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Focus on 'Close' prices
    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)

    # Forward-fill missing values
    df.ffill(inplace=True)

    # Normalize prices with individual scalers
    scaler = MinMaxScaler()
    df['normalized'] = scaler.fit_transform(df[['price']])

    # Store
    ticker_data[ticker] = df

print(f"✅ Loaded {len(ticker_data)} tickers.")

# To perform time series forecasting, we convert the normalized series into input/output sequences 
# using a sliding window approach.
# [past inputs] → [future targets]
def create_sequences(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# Initialize holders for aggregated data
X_train_all, y_train_all = [], []
X_test_all, y_test_all = [], []

# Apply to each ticker
for ticker, df in ticker_data.items():
    print(f"🔄 Creating sequences for {ticker}...")
    series = df['normalized'].values
    X, y = create_sequences(series, WINDOW_SIZE)

    # Time-respecting 80/20 split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Append to global containers
    X_train_all.append(X_train)
    y_train_all.append(y_train)
    X_test_all.append(X_test)
    y_test_all.append(y_test)

    print(f"✅ {ticker}: Train={X_train.shape}, Test={X_test.shape}")

# Combine across all tickers
X_train = np.vstack(X_train_all)
y_train = np.hstack(y_train_all)
X_test = np.vstack(X_test_all)
y_test = np.hstack(y_test_all)

print(f"\n📦 Combined Dataset — Train: {X_train.shape}, Test: {X_test.shape}")



# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # Shape: (batch, seq, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# Create combinations
hyperparam_combinations = list(itertools.product(n_qubit_options, q_depth_options, n_rot_params_options))

# Iterate over each combination
for n_qubits, q_depth, n_rot_params in hyperparam_combinations:
    if use_quantum :
        added_notes = experiment_notes+f"\nGridSearch | Qubits={n_qubits}, Depth={q_depth}, RotParams={n_rot_params}"

        
        print(f"\n🔍 Experimenting with: {experiment_notes}"
            f"\nQubits: {n_qubits}, Depth: {q_depth}, RotParams: {n_rot_params}")
        
    train_and_evaluate(
        y_test_tensor=y_test_tensor,
        train_loader=train_loader,
        test_loader=test_loader,
        X_train=X_train,
        scaler=scaler,
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        q_depth=q_depth,
        n_rot_params=n_rot_params,
        ML_unit_type=ML_unit_type,
        num_ML_layers=num_ML_layers,
        use_quantum=use_quantum,
        post_quantum_activation=post_quantum_activation,
        skip_connection=skip_connection,
        output_activation=output_activation,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        use_layernorm=use_layernorm,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        learning_rate=learning_rate,
        tickers=tickers,
        window_size=WINDOW_SIZE,
        notes=added_notes
    )

