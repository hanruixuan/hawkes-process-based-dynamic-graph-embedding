# Hawkes process-based time-series graph/network representation learning (Embedding)
This repository utilizes Hawkes process to sim to learn time series graph embedding. 
- This repo utilizes Hawkes process to simulate the network evolving process.
- The (`preprocess.py`) converts the raw dataset into time series networks. Each represents a network structure within a period of time (i.e. 7 days).
- The (`train.py`) uses (`load_data.py`) to load time series networks and embeds them into the vector space. The embedding will be stored (`/emb`) directory
- The Python environment is Python 3.6 or higher and needs PyTorch and NumPy libraries.
