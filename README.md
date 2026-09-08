# 🧅 News-Enhanced Multi-Target Onion Price Forecasting System

An end-to-end Deep Learning & NLP forecasting system for agricultural commodities, specifically predicting **Minimum**, **Modal (Average)**, and **Maximum** prices (₹/Quintal) for onion trading at **APMC Pimpalgaon Baswant (Nashik District, Maharashtra)** — Asia's largest onion market hub.

---

## 📌 Key Highlights

- **Multi-Target Forecasting**: Predicts daily Minimum, Modal, and High prices simultaneously using a Stacked Long Short-Term Memory (LSTM) network.
- **Multimodal Feature Fusion**: Combines historical APMC arrivals/prices, IMD Pune/Nashik meteorological data (rainfall, humidity, temperature), and economic news sentiment.
- **Time-Aware News Sentiment Engine**: Scrapes and analyzes agricultural market news (PIB, NAFED, Agrowon, APMC updates) with strict zero future-leakage ($t_{\text{pub}} \le t-1$) and exponential time-decay weighting.
- **Live Scrapers & Weather Integrations**: Real-time rate scraping from `apmcpimpalgaon.com` and live IMD weather tracking.
- **Interactive Streamlit Dashboard**: User-friendly UI with multi-trace Plotly charts, dynamic lookback selection (7 vs 14 days), forecast horizons (7, 14, 30, 90 days), and scenario simulations.

---

## 🏗️ System Architecture

```text
                                  [ RAW DATA INGESTION ]
                                             │
      ┌────────────────────────┬─────────────┴───────────────┬────────────────────────┐
      ▼                        ▼                             ▼                        ▼
Historical APMC Records  IMD Weather Normals           Historical News/Events    Live Scrapers
(Daily Mandi Arrivals)   (Rain, Temp, Humidity)        (PIB, NAFED, Agrowon)    (APMC Portal & IMD)
      │                        │                             │                        │
      └────────────────────────┼─────────────────────────────┴────────────────────────┘
                               ▼
                 [ src/data_processor.py ]
                 • Daily arrival-weighted aggregation
                 • Continuous calendar alignment & forward filling
                 • Technical features (Lag1, 3D MA, Spreads)
                               │
                               ▼
                 [ src/news_processor.py ]
                 • News deduplication & token normalization
                 • 9 economic sentiment signals [-1.0, +1.0]
                 • Zero-leakage constraint & exponential time-decay
                               │
                               ▼
                 [ src/dataset_prep.py ]
                 • 24-feature consolidated matrix
                 • Chronological train/validation/test split
                 • Decoupled dual scalers (Features & Targets)
                 • Sequence sliding window generator
                               │
                               ▼
                 [ src/lstm_model.py ]
                 • Stacked Multi-Target LSTM (Keras 3 + PyTorch)
                 • Huber Loss & Adam Optimization
                 • Walk-forward iterative multi-step prediction
                               │
                               ▼
                 [ app.py / main.py ]
                 • Streamlit Interactive Web Application
                 • Real-time forecasts & interactive charts
```

---

## 📂 Repository Structure

```text
├── app.py                      # Streamlit interactive web dashboard
├── main.py                     # CLI pipeline runner for end-to-end model training/prediction
├── evaluate_metrics.py         # Comprehensive evaluation metrics & regression analytics
├── pyrightconfig.json          # Python language server configuration
├── requirements.txt            # Project dependencies
├── PROJECT_DOCUMENTATION.md    # Detailed academic/technical documentation
├── models/                     # Saved pre-trained models and scalers
│   ├── best_lstm_model.keras   # High-accuracy stacked LSTM model artifact
│   ├── scaler.pkl              # Feature scaler
│   └── scaler_price.pkl        # Price target scaler
└── src/                        # Core modular source code
    ├── apmc_scraper.py         # Real-time APMC auction scraper
    ├── data_processor.py       # Data cleaning, calendar filling, feature engineering
    ├── dataset_prep.py         # Windowing, scaling, and dataset generation
    ├── lstm_model.py           # Stacked LSTM architecture and recursive forecasting
    ├── news_processor.py       # NLP news sentiment extraction & aggregation
    └── weather_fetcher.py      # Real-time and historical IMD weather integration
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Siddhant8010/price-forecasting.git
cd price-forecasting
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Run the CLI Pipeline
```bash
python main.py
```

---

## 📊 Evaluation & Performance

The model predicts three price dimensions simultaneously:
- **Minimum Price (₹/Qtl)**
- **Modal Price (₹/Qtl)**
- **Maximum / High Price (₹/Qtl)**

Key performance metrics evaluated across test splits:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy (DA) & $R^2$ Score

---

## 📜 License & Acknowledgments

- **Target Market**: APMC Pimpalgaon Baswant, Nashik, Maharashtra
- **Data Sources**: APMC Pimpalgaon Baswant, India Meteorological Department (IMD), Press Information Bureau (PIB), Agrowon.
