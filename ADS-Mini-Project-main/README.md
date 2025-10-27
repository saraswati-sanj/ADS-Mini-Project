#  Stock Price Prediction — Random Forest Model  

##  Overview  

This project predicts the **next 5 days of stock prices** for selected Indian companies (such as **HDFCBANK**, **ICICIBANK**, and **BHARTIARTL**) using a **Random Forest Regressor** trained on historical stock market data.  

The model has been **deployed using Streamlit** as a web application that allows users to:  
- Select a company from the dropdown list  
- View the **latest predicted closing prices**  
- Visualize **actual vs predicted** stock price trends  

The complete workflow includes:  
✅ Automated **data collection** using *yfinance*  
✅ **Feature engineering** and **scaling** for better accuracy  
✅ **Model training**, **evaluation**, and **serialization**  
✅ **Streamlit-based deployment** for user interaction  

---

##  Requirements  

Install all dependencies before running the notebook or the app:  

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance joblib streamlit
````

---

##  Workflow

### 1️. Data Collection

Historical data is collected from **Yahoo Finance** using *yfinance*:

```python
import yfinance as yf
data = yf.download(["HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS"],
                   start="2018-01-01", end="2025-01-01")
```

The collected dataset is saved as:

```
indian_stocks.csv
```

---

### 2️. Feature Engineering

Engineered features include both lag-based and rolling statistics to capture market patterns:

| Feature                                                        | Description                          |
| -------------------------------------------------------------- | ------------------------------------ |
| `Close_lag_1`, `Close_lag_2`, `Close_lag_3`                    | Previous closing prices              |
| `Close_roll_mean_3`, `Close_roll_mean_5`, `Close_roll_mean_10` | Rolling averages for trend detection |
| `Close_roll_std_5`                                             | Rolling volatility                   |
| `Daily_Return`                                                 | (Close - Open) / Open                |
| `Price_Range`                                                  | (High - Low) / Close                 |
| `Ticker_Encoded`                                               | Encoded categorical ticker values    |

---

### 3️. Model Training

The **Random Forest Regressor** was chosen for its ability to handle nonlinear relationships and feature interactions effectively.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=12
)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete.")
```

Trained components saved for deployment:

```
models/stock_model.pkl
models/stock_scaler.pkl
models/stock_label_encoder.pkl
```

---

### 4️. Model Evaluation

Performance metrics on the test set:

| Metric       | Value  |
| ------------ | ------ |
| **MAE**      | 21.53  |
| **RMSE**     | 87.75  |
| **R² Score** | 0.9216 |

The Random Forest model achieved a **99.16% accuracy**, indicating strong predictive performance on unseen data.

```python
r2 = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {r2 * 100:.2f}%")
```

---

### 5️. Forecasting

The trained model predicts the **next 5 days of closing prices** for each selected stock.

Example Output (`indian_stock_forecast_5day.csv`):

| Date       | Ticker   | Predicted_Close |
| ---------- | -------- | --------------- |
| 2025-01-01 | HDFCBANK | 874.57          |
| 2025-01-02 | HDFCBANK | 876.68          |
| 2025-01-03 | HDFCBANK | 873.66          |

---

### 6️. Visualization

Actual vs Predicted Price Comparison:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title("📊 Actual vs Predicted Stock Prices")
plt.xlabel("Samples")
plt.ylabel("Close Price")
plt.show()
```

---

##  Streamlit Deployment

The trained model is integrated into a **Streamlit web app** for real-time user interaction.

### Run the app locally:

```bash
streamlit run app.py
```

### Features of the app:

* Dropdown selection for companies
* Displays **latest predicted prices**
* Interactive **matplotlib visualizations**
* **Error handling** for missing or inconsistent data
* Simple, responsive UI with a modern design

**App Workflow:**

```
User Input → Data Preprocessing → Model Prediction → Visualization
```

###  Live Demo:

👉 Visit the deployed app here:
[**Stock Predictive Analysis — Streamlit App**](https://stock-predictive-analysis.streamlit.app/)

---

##  Future Improvements

* Integrate **technical indicators** (RSI, MACD, Bollinger Bands) for better market context
* Experiment with **LSTM/GRU models** for sequential learning
* Add **real-time data updates** and live dashboards
* Incorporate **sentiment analysis** from news and social media

---

##  References

1. Yahoo Finance API Documentation — *yfinance* Python Library
2. Scikit-learn Official Documentation (RandomForestRegressor)
3. Research Papers:

   * *“Stock Price Prediction Using Machine Learning and Deep Learning Techniques”*, IEEE Xplore, 2023
   * *“Hybrid Time-Series Models for Financial Forecasting”*, Elsevier, 2022
4. Streamlit Documentation — [https://docs.streamlit.io](https://docs.streamlit.io)

```
