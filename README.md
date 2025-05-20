
# 🔮 Cryptocurrency Price Prediction API (Backend)

This repository contains the **backend API** for the **Cryptocurrency Price Prediction System** using a **Long Short-Term Memory (LSTM)** model. Built using **FastAPI** (preferred for async operations) or **Flask**, the backend handles:

- Real-time and historical crypto data fetching  
- Preprocessing for time-series modeling  
- LSTM-based price prediction  
- RESTful API endpoints for frontend integration  
- PostgreSQL (Supabase) database integration



## 📂 Project Structure


├── app/
│   ├── main.py               # Entry point (FastAPI/Flask server)
│   ├── models.py             # LSTM model loading & inference
│   ├── data_handler.py       # Data fetching & preprocessing (yfinance, ccxt)
│   ├── routes/
│   │   ├── prediction.py     # /predict endpoint
│   │   └── data.py           # /fetch endpoint
│   ├── utils/
│   │   └── scaler.py         # Normalization & inverse transforms
├── requirements.txt
├── Dockerfile
└── README.md




## ⚙️ Tech Stack

* **Backend Framework**: FastAPI / Flask
* **Machine Learning**: TensorFlow, Keras, scikit-learn
* **Data Handling**: yfinance, ccxt, pandas, numpy
* **Database**: PostgreSQL (via Supabase)
* **Deployment**: Docker, TensorFlow Serving, GitHub Actions (CI/CD)



## 🔌 API Endpoints

### `POST /predict`

Predict future cryptocurrency prices using the trained LSTM model.

#### Request:

```json
{
  "symbol": "BTC-USD",
  "lookback_days": 60
}
```

#### Response:

```json
{
  "predicted_price": 43892.55,
  "date": "2025-05-22"
}
```

---

### `GET /fetch?symbol=BTC-USD`

Fetch historical crypto price data for training and analysis.

#### Response:

```json
{
  "symbol": "BTC-USD",
  "historical_data": [...]
}
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crypto-lstm-backend.git
cd crypto-lstm-backend
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI server (recommended)

```bash
uvicorn app.main:app --reload
```

Or run Flask server if applicable:

```bash
python app/main.py
```

---

## 🧠 Model Training (Optional)

Training is typically done offline and the trained model is loaded via `models.py`.

* Use historical data (via `yfinance` or `ccxt`)
* Normalize using `MinMaxScaler`
* Train and save `.h5` model file
* Load during runtime for predictions

---

## 🐳 Docker Support

### Build Docker Image

```bash
docker build -t crypto-lstm-backend .
```

### Run Container

```bash
docker run -d -p 8000:8000 crypto-lstm-backend
```

---

## 🧪 Testing

Unit and integration testing (optional) can be added via `pytest` or `unittest`.

---

## 🗃️ Database (Supabase)

* **Tables**:

  * `users` – (for login/auth system if implemented)
  * `predictions` – (cache past predictions)
  * `historical_data` – (store fetched crypto price data)

---

## 📌 To-Do (for future enhancements)

* [ ] Add JWT-based authentication
* [ ] Multi-crypto support
* [ ] Integrate with Trading Bot API
* [ ] Enable scheduled model retraining via cron/GitHub Actions

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* TensorFlow & Keras for deep learning
* FastAPI for building performant APIs
* Supabase for managed PostgreSQL with real-time features
* yFinance & CCXT for historical crypto data

---


