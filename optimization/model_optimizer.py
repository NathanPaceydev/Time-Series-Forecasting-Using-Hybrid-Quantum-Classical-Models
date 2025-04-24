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
import os
import random
import re

from sklearn.metrics import mean_absolute_error, mean_squared_error




    
    



def train_and_evaluate( y_test_tensor,train_loader, test_loader, X_train, scaler, input_size, 
                       output_size, hidden_size, n_qubits, q_depth, n_rot_params, ML_unit_type, 
                       num_ML_layers, use_quantum, post_quantum_activation, skip_connection,
                       output_activation, use_dropout, dropout_rate, use_layernorm, num_epochs, 
                       early_stop_patience, learning_rate, tickers, window_size, notes
):
    # Device definition
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def quantum_circuit(inputs, weights):
        """
        Variational quantum circuit for hybrid model.
        
        Args:
            inputs (Tensor): Input features (size ≤ n_qubits)
            weights (Tensor): Trainable parameters of shape (q_depth, n_qubits, n_rot_params)
            draw (bool): If True, draw the circuit once (ASCII + matplotlib)
            
        Returns:
            List[Expectation values] for PauliZ on each qubit
        """
        # --- Input Encoding ---
        encoded_inputs = inputs[:n_qubits]
        for i in range(n_qubits):
            qml.RY(encoded_inputs[i], wires=i)
        
        # Optionally, use this instead:
        # qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

        # --- Variational Layers ---
        for layer in range(q_depth):
            for i in range(n_qubits):
                if n_rot_params == 1:
                    qml.RY(weights[layer][i][0], wires=i)
                elif n_rot_params == 2:
                    qml.RX(weights[layer][i][0], wires=i)
                    qml.RZ(weights[layer][i][1], wires=i)
                elif n_rot_params == 3:
                    qml.Rot(*weights[layer][i], wires=i)
                else:
                    raise ValueError("n_rot_params must be 1, 2, or 3.")

            # Entanglement (ring topology)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        

        # --- Measurement ---
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    class QuantumLayer(nn.Module):
        def __init__(self, n_qubits, q_depth, n_rot_params):
            super().__init__()

            # Initialize trainable parameters for the quantum circuit
            # Shape: [q_depth, n_qubits, n_rot_params] (rotation angles per qubit per layer)
            q_init = torch.empty(q_depth, n_qubits, n_rot_params)
            torch.nn.init.normal_(q_init, mean=0.0, std=0.01)  # Small init to avoid flat gradients
            self.q_params = nn.Parameter(q_init)

            self.n_qubits = n_qubits
            self.q_depth = q_depth
            self.n_rot_params = n_rot_params

        def forward(self, x_batch):
            """
            Apply the quantum circuit to each sample in the batch.

            Args:
                x_batch (Tensor): Input of shape [batch_size, n_qubits]

            Returns:
                Tensor: Output of shape [batch_size, n_qubits]
            """
            outputs = []
            for x in x_batch:
                # Apply the quantum circuit to each sample
                q_out = quantum_circuit(x, self.q_params)

                # Convert list of expectation values into a float32 tensor
                q_tensor = torch.stack(q_out).to(dtype=torch.float32)

                outputs.append(q_tensor)

            # Stack the results into a batch tensor
            return torch.stack(outputs)
    
    class HybridQNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size,
                    n_qubits=4, q_depth=1, n_rot_params=3, ML_unit_type = 'LSTM', num_ML_layers = 1, use_quantum=True,
                    post_quantum_activation=None, skip_connection="concat",output_activation="Sigmoid",
                    use_dropout=False, dropout_rate=0.3,
                    use_layernorm=False):
            super().__init__()
            
            self.use_quantum = use_quantum
            self.post_quantum_activation = post_quantum_activation  # 👈 Store this
            self.skip_connection = skip_connection  # 👈 Store this
            self.n_qubits = n_qubits  # 👈 Store this
            self.use_dropout = use_dropout
            self.dropout_rate = dropout_rate
            self.use_layernorm = use_layernorm



            ML_unit_type = ML_unit_type.upper()
            # ✅ Check that it's at least valid before trying to build
            if not any(unit in ML_unit_type for unit in ["LSTM", "GRU", "RNN"]):
                raise ValueError(f"Unsupported ML unit type: {ML_unit_type}. Supported types: LSTM, GRU, RNN")

            # Convert String to list of units
            #sanitized_units = re.sub(r"[^a-zA-Z]", "-", ML_unit_type.upper())

            # Example: ML_unit_type = "LSTM-GRU-LSTM"
            unit_sequence = [u.strip().upper() for u in re.split(r"[^a-zA-Z0-9]+", ML_unit_type) if u.strip()]

            # Build the sequence stack
            self.sequence_stack = nn.Sequential()

            for idx, unit in enumerate(unit_sequence):
                layers = num_ML_layers if idx == 0 else 1  # Only first gets full depth
                if unit == "LSTM":
                    self.sequence_stack.add_module(f"lstm_{idx}", nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                                                        num_layers=layers, batch_first=True))
                    input_size = hidden_size  # Update input size for next unit
                elif unit == "GRU":
                    self.sequence_stack.add_module(f"gru_{idx}", nn.GRU(input_size=input_size, hidden_size=hidden_size,
                                                                        num_layers=layers, batch_first=True))
                    input_size = hidden_size
                elif unit == "RNN":
                    self.sequence_stack.add_module(f"rnn_{idx}", nn.RNN(input_size=input_size, hidden_size=hidden_size,
                                                                        num_layers=layers, batch_first=True))
                    input_size = hidden_size
                else:
                    raise ValueError(f"Unsupported unit: {unit}")
            

            # Linear layer to map LSTM output to quantum layer input size
            self.linear1 = nn.Linear(hidden_size, n_qubits)

            # Optional quantum layer for variational feature extraction
            if use_quantum:
                self.quantum = QuantumLayer(n_qubits=n_qubits, q_depth=q_depth, n_rot_params=n_rot_params)

            if self.use_layernorm:
                self.layernorm = nn.LayerNorm(n_qubits)

            if self.use_dropout:
                self.dropout = nn.Dropout(p=self.dropout_rate)
        
            # ⬇ decide how many features go into final layer
            if skip_connection == "concat":
                in_features = 2 * n_qubits
            else:
                in_features = n_qubits

            # Final linear layer
            self.linear2 = nn.Linear(in_features, output_size)

            # Final activation function for output layer
            # to any of the supported activation functions
            if output_activation == "ReLU":
                self.output_activation = nn.ReLU()
            elif output_activation == "Tanh":
                self.output_activation = nn.Tanh()
            elif output_activation == "Sigmoid":
                self.output_activation = nn.Sigmoid()
            elif output_activation == "Softmax":
                self.output_activation = nn.Softmax(dim=1)
            else:
                self.output_activation = None
                
            self.output_activation_name = output_activation  # ← This keeps the original string

            
        def forward(self, x):
            # Input shape check: should be (batch_size, sequence_length, input_size)
            if len(x.shape) != 3:
                raise ValueError(f"Expected input shape (batch_size, seq_len, input_size), got {x.shape}")

            # Run input through sequence stack (LSTM/GRU/RNN)
            rnn_out = x
            for layer in self.sequence_stack:
                if isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)):
                    rnn_out, _ = layer(rnn_out)
                else:
                    rnn_out = layer(rnn_out)


            # Use the final hidden state of the sequence
            last_hidden = rnn_out[:, -1, :]

            # Project to quantum input dimension
            reduced = self.linear1(last_hidden)

            # Apply quantum layer if enabled
            if self.use_quantum:
                quantum_out = self.quantum(reduced)
                
                if self.post_quantum_activation == "ReLU":
                    quantum_out = torch.relu(quantum_out)
                    
                elif self.post_quantum_activation == "Tanh":
                    quantum_out = torch.tanh(quantum_out)
                    
                elif self.post_quantum_activation == "Sigmoid":
                    quantum_out = torch.sigmoid(quantum_out)
            
            else:
                quantum_out = reduced  # fallback to classical-only path
                
            # ✅ Optional regularization after quantum
            if self.use_layernorm:
                quantum_out = self.layernorm(quantum_out)

            if self.use_dropout:
                quantum_out = self.dropout(quantum_out)


            # Concatenate classical and quantum outputs
            if self.skip_connection == "add":
                x_out = quantum_out + reduced
            elif self.skip_connection == "concat":
                x_out = torch.cat([quantum_out, reduced], dim=1)
            else:
                x_out = quantum_out  # quantum-only path

            

            # Final linear projection
            out = self.linear2(x_out)

            return self.output_activation(out) if self.output_activation else out


    
    # Set the sequence length based on the training data
    sequence_length = X_train.shape[1]

    # Instantiate the model 
    # (use_quantum=True for quantum model, False for classical model)
    model = HybridQNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_qubits=n_qubits,
        q_depth=q_depth,
        n_rot_params=n_rot_params,
        ML_unit_type=ML_unit_type,
        num_ML_layers=num_ML_layers,
        use_quantum=use_quantum,  # Set to True for quantum model
        post_quantum_activation=post_quantum_activation,
        skip_connection=skip_connection,
        output_activation=output_activation,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        use_layernorm=use_layernorm
    )

    #use GPU if available **TODO**
        
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        # Training loop
    best_val_loss = float('inf')
    # Use tqdm for progress bar in each epoch
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Use tqdm for progress bar in each epoch
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for X_batch, y_batch in train_loader_tqdm:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"📘 Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # === Early Stopping Check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optional: save model checkpoint
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= early_stop_patience:
            print("⛔ Early stopping triggered.")
            break
        
        
    # === Load the best model for testing/prediction ===
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    print("✅ Loaded best model from checkpoint for final evaluation.")

    predictions = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.detach().cpu())

    # Concatenate and flatten
    predictions = torch.cat(predictions).squeeze().numpy()
    y_true = y_test_tensor.squeeze().cpu().numpy()
    
    # Reshape for inverse_transform (must be 2D)
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_prices = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    
    
    mae = mean_absolute_error(true_prices, predicted_prices)
    rmse = mean_squared_error(true_prices, predicted_prices, squared=False)

    print(f"📈 Test MAE  = {mae:.4f}")
    print(f"📈 Test RMSE = {rmse:.4f}")

    
    # Compute percentage errors
    percentage_errors = 100 * (predicted_prices - true_prices) / true_prices

    avg_percent_error = np.mean(np.abs(percentage_errors))
    print(f"Average Absolute Percentage Error: {avg_percent_error:.2f}%")
    
    def log_experiment(model, window_size, num_epochs, patience,
                   avg_train_loss, avg_val_loss, mae, rmse,
                   avg_percent_error, tickers, notes=""):

        # --- Extract model config ---
        use_quantum = getattr(model, "use_quantum", False)
        post_quantum_activation = getattr(model, "post_quantum_activation", None)
        skip_connection = getattr(model, "skip_connection", None)
        final_activation = getattr(model, "output_activation_name", None)
        use_dropout = getattr(model, "use_dropout", False)
        dropout_rate = getattr(model, "dropout_rate", None)
        use_layernorm = getattr(model, "use_layernorm", False)

        # --- Sequence architecture inspection ---
        lstm_layers = 0
        gru_layers = 0
        rnn_layers = 0
        hidden_size = None

        for module in model.sequence_stack:
            if isinstance(module, nn.LSTM):
                lstm_layers += module.num_layers
                hidden_size = module.hidden_size
            elif isinstance(module, nn.GRU):
                gru_layers += module.num_layers
                hidden_size = module.hidden_size
            elif isinstance(module, nn.RNN):
                rnn_layers += module.num_layers
                hidden_size = module.hidden_size

        # --- Quantum circuit info (if used) ---
        if use_quantum and hasattr(model, "quantum"):
            n_qubits = getattr(model.quantum, "n_qubits", "-")
            q_depth = getattr(model.quantum, "q_depth", "-")
            n_rot_params = getattr(model.quantum, "n_rot_params", "-")
        else:
            n_qubits, q_depth, n_rot_params = "-", "-", "-"

        # --- Load or initialize Excel sheet ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, "qml_experiment_log.xlsx")
        
        try:
            df = pd.read_excel(log_path)
            next_id = int(df["Experiment ID"].max()) + 1
        except (FileNotFoundError, ValueError, KeyError):
            df = pd.DataFrame()
            next_id = 1

        # --- Assemble experiment entry ---
        new_result = {
            # 🧾 Metadata
            "Experiment ID": next_id,
            "Tickers": ", ".join(tickers),
            "Description": f"Auto-log: Q={n_qubits}, D={q_depth}, Skip={skip_connection}",

            # 🧠 Classical architecture
            "LSTM Layers": lstm_layers,
            "GRU Layers": gru_layers,
            "RNN Layers": rnn_layers,
            "Hidden Size": hidden_size,
            "Window Size": window_size,

            # ⚛️ Quantum architecture
            "Use Quantum": use_quantum,
            "Qubits": n_qubits,
            "Q Depth": q_depth,
            "Rotation Params": n_rot_params,
            "Skip Connection": skip_connection,
            "Post-Quantum Activation": post_quantum_activation,

            # 🎛️ Regularization & Activation
            "Use Dropout": use_dropout,
            "Dropout Rate": dropout_rate,
            "Use LayerNorm": use_layernorm,
            "Final Activation": final_activation,

            # 🏋️ Training setup
            "Num Epochs": num_epochs,
            "Early Stop Patience": patience,

            # 📈 Results
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "MAE": mae,
            "RMSE": rmse,
            "Avg % Error": avg_percent_error,

            # 🗒️ Notes
            "Notes": notes
        }

        # --- Append and save ---
        df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)
        

        try:
            df.to_excel(log_path, index=False)
            print(f"📋 Experiment logged as ID {next_id}")
            print("📂 Log file saved at:", os.path.abspath(log_path))
        except Exception as e:
            print(f"❌ Failed to write Excel log: {e}")
        
           
        
    log_experiment(
        model=model,
        window_size=window_size,
        num_epochs=num_epochs,
        patience=early_stop_patience,
        avg_train_loss=avg_train_loss,
        avg_val_loss=avg_val_loss,
        mae=mae,
        rmse=rmse,
        avg_percent_error=avg_percent_error,
        tickers=tickers,
        notes=notes
    )



    return None 
