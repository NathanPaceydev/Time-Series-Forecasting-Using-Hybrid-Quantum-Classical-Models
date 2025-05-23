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
n_qubits = 6
q_depth = 4
n_rot_params = 3

# wether to use quantum or classical model
use_quantum = False  # True or False

#---------- classical model parameters ----------


# === Model Config ===
# Set model hyperparameters
input_size = 1
hidden_size = 32
output_size = 1

# # Default values for classical model
learning_rate = 0.001

num_epochs = 10
use_dropout=False
dropout_rate= 0.0 # 0.2 to 0.5 is common for LSTM/GRU
use_layernorm=False 

# === Early Stopping Config ===
early_stop_patience = 2 # Number of epochs to wait for improvement
epochs_no_improve = 0

# Type of ML unit to use
ML_unit_type = "GRU"  # "LSTM", "GRU", or "RNN" can havy many layers
num_ML_layers = 2  # Number of LSTM/GRU/RNN layers

post_quantum_activation = "Tanh"  # "ReLU", "Tanh", "Sigmoid", or None
skip_connection = "add"  # "concat" or "add"
output_activation = None # "ReLU", "Tanh", "Sigmoid", "Softmax", or None

#----------------- Ranges --------------------------------
# === Classical Parameters to Explore ===
hidden_size_options = [8, 16, 32, 64]
# Top 3 best performing dropout rates based on low Avg % Error (from user data)
best_dropouts = [0.0, 0.16, 0.59]

# Define fine-grained ranges around each of the best performing values
fine_tuned_dropouts = []

for d in best_dropouts:
    # Generate 5 values in a small range around the best dropout, avoiding values below 0 or above 1
    lower = max(0.0, d - 0.05)
    upper = min(1.0, d + 0.05)
    fine_range = np.round(np.linspace(lower, upper, 10), 4)
    fine_tuned_dropouts.extend(fine_range)

# Remove any duplicates and sort
unique_fine_tuned_dropouts = sorted(set(fine_tuned_dropouts))
dropout_rate_options = unique_fine_tuned_dropouts
num_layers_options = [1, 2, 3, 4, 5, 6, 7, 8]  # Number of LSTM/GRU/RNN layers
ml_unit_options = ["LSTM", "GRU", "LSTM + GRU", "RNN", "LSTM + RNN", "GRU + RNN", "GRU + LSTM", "LSTM + GRU + RNN"]
skip_connection_options = ["concat", "add"]
activation_options = [None, "ReLU", "Tanh", "Sigmoid"]
activation_combinations = list(itertools.product(activation_options, activation_options))


# === Choose the parameter you want to optimize ===
param_to_optimize = 'num_layers'  # 👈 Change this to any: "dropout_rate", "num_layers", etc.

# === Define a mapping to loop over options ===
param_grid = {
    "hidden_size": hidden_size_options,
    "dropout_rate": dropout_rate_options,
    "num_layers": num_layers_options,
    "ML_unit": ml_unit_options,
    "skip_connection": skip_connection_options,
    "output_activation": activation_options,
}



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



added_notes = "Classical Opt: with default values"
# === Loop over the selected parameter ===
if param_to_optimize == "output_activation":
    for post_act, out_act in activation_combinations:
        post_quantum_activation = post_act
        output_activation = out_act
        added_notes = f"Classical Opt: with post_activation={post_act}, output_activation={out_act}"
        print(f"\n🧪 Optimizing Activations → Post: {post_act}, Output: {out_act}")

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

elif param_to_optimize:
    for value in param_grid[param_to_optimize]:    
        # Then override the one we’re testing
        if param_to_optimize == "hidden_size":
            hidden_size = value
        elif param_to_optimize == "dropout_rate":
            dropout_rate = value
            use_dropout = value > 0
        elif param_to_optimize == "num_layers":
            num_ML_layers = value
        elif param_to_optimize == "ML_unit":
            ML_unit_type = value
        elif param_to_optimize == "skip_connection":
            skip_connection = value

        added_notes = f"Classical Opt: with {param_to_optimize}={value}"

        print(f"\n🧪 Optimizing {param_to_optimize} → {value}")

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
else:
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
