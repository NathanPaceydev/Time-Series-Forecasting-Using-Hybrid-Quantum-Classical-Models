# Hybrid Quantum-Classical Neural Networks for Stock Price Forecasting

This project explores hybrid quantum-classical machine learning models for time series forecasting, using historical stock data as a challenging real-world benchmark. The architecture, training pipeline, and evaluation framework are fully modular, enabling rapid experimentation with classical and quantum components.

## 📁 Repository Structure

| Folder/File                        | Description |
|-----------------------------------|-------------|
| `model/`                          | Final modular implementation of the forecasting model, including both quantum and classical variants. |
| `optimization/`                   | Python scripts and logs used for sweeping hyperparameters and evaluating performance across architectures. |
| `Rough work/`                     | Notes, debugging experiments, early architecture sketches, and analysis of model challenges. |
| `Thesis American Style Stock options/` | Reference material and previous work used to inform the model design, from a bachelor’s thesis on option forecasting. |
| `images/`                         | Visualizations of model architecture, training curves, and quantum circuit configurations. |
| `Report/`                       | Project Documentation including a Report and Presentation Slides|

## 📈 Project Summary

This repository implements a hybrid forecasting model that combines:

- **Classical recurrent units** (LSTM, GRU, RNN)
- **Optional variational quantum circuit** layers via [PennyLane](https://pennylane.ai/)
- A dynamic architecture that supports configuration of:
  - Recurrent unit types and stacking order
  - Quantum layer inclusion, qubit count, circuit depth, and gate structure
  - Regularization (dropout, layer norm)
  - Skip connections and activation functions

The task was framed around forecasting normalized daily closing prices of AAPL, MSFT, and GOOGL using sliding windows of historical data.

## 🔬 Methodology Highlights

- **Data Handling:**  
  Historical stock data was fetched using the `yfinance` API (2015–2024). Prices were normalized per ticker using Min-Max scaling and transformed into time series sequences using a sliding window of 20 days. An 80/20 split ensured chronological integrity.

- **Architecture:**  
  The network is defined in `HybridQNN`, a PyTorch model that supports stacked recurrent layers and optional quantum circuits. Quantum inputs are linearly projected from the final hidden state and processed via a `qml.QNode`. Both classical and quantum outputs are merged via concatenation or addition.

- **Training:**  
  Models were trained using MSE loss and the Adam optimizer. Early stopping was applied (patience: 2–4 epochs). Performance metrics include MAE, RMSE, and average percentage error.

- **Optimization:**  
  Hyperparameter tuning was conducted classically first (dropout, activation, hidden size, layers, RNN types) to reduce quantum execution time. Sweeps were implemented in dedicated scripts under `optimization/`.
