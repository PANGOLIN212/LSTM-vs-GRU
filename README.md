# Air Passengers Forecasting: LSTM vs GRU

A time series forecasting project that compares two recurrent neural network architectures, **LSTM** and **GRU**, on the classic Air Passengers dataset (monthly international airline passengers, 1949–1960).

## Short Description

> Forecasting monthly airline passenger numbers with LSTM and GRU neural networks in Keras, and comparing their accuracy using MAE, RMSE, and MAPE. In this experiment, the GRU model outperformed the LSTM on the held-out test set.

## Dataset

- **Source:** Air Passengers dataset (`AirPassengers.csv`), a widely used time series benchmark
- **Size:** 144 monthly observations (January 1949 – December 1960)
- **Columns:** `Month`, `#Passengers`
- **Characteristics:** clear upward trend with growing yearly seasonality

## Methodology

1. **Load and explore** the data, parse dates, check for missing values, and plot the series
2. **Scale** passenger counts to the range [0, 1] with `MinMaxScaler`
3. **Split chronologically:** 80% train (115 months), 20% test (29 months)
4. **Create sequences** using a sliding window of 12 months to predict the next month
5. **Train two models** with identical settings so the comparison is fair:
   - LSTM: `LSTM(50)` → `Dense(1)`
   - GRU: `GRU(50)` → `Dense(1)`
   - Adam optimizer, MSE loss, 100 epochs, batch size 8
6. **Inverse-transform** the predictions back to passenger counts and evaluate

## Results

| Model | MAE   | RMSE  | MAPE (%) |
|-------|-------|-------|----------|
| LSTM  | 66.75 | 77.94 | 13.96    |
| GRU   | 37.89 | 47.14 | 7.88     |

The GRU achieved roughly half the error of the LSTM on this test set, with fewer parameters (8,001 vs 10,451).

**Caveats:** The test set is small (17 sequences), the test period includes the highest-traffic years, and the LSTM's validation loss was noisy and rising toward the end of training, which suggests some overfitting. Results may vary between runs because of random weight initialization, so treat the ranking as indicative rather than definitive.

## Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib
- scikit-learn
- TensorFlow / Keras

## Getting Started

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Run

1. Download the Air Passengers CSV and place it in the project folder
2. Update the file path in the notebook's data loading cell, for example:
   ```python
   df = pd.read_csv("AirPassengers.csv")
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Untitled5.ipynb
   ```

## Project Structure

```
.
├── Untitled5.ipynb     # Data prep, model training, evaluation, plots
├── AirPassengers.csv   # Dataset (add this yourself)
└── README.md
```

## Possible Improvements

- Add early stopping and dropout to reduce overfitting
- Run multiple seeds and report the average and spread of each metric
- Use a seasonal naive baseline or ARIMA/SARIMA for comparison
- Try stacked layers, different window sizes, or hyperparameter tuning
- Forecast multiple steps ahead rather than one month at a time
- Use a differenced or log-transformed series to handle the trend

## License

This project is open for learning and educational use.
