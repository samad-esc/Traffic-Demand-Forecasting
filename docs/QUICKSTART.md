# 🚀 Quick Start Guide

Get your Traffic Demand Forecasting application running in 5 minutes!

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Train the Models (5-10 minutes)

```bash
python train.py
```

**What happens:**
- ✅ Loads train.csv and test.csv from `data/` folder
- ✅ Engineers all features from your notebook
- ✅ Trains LightGBM model
- ✅ Trains CatBoost model
- ✅ Saves models to `models/` folder
- ✅ Displays evaluation metrics

**Expected output at the end:**
```
FINAL RESULTS:
================================================================================
Ensemble RMSE: 0.136XXX
Ensemble MAE: 0.091XXX
Ensemble R² Score: 0.71XX
================================================================================
```

## Step 3: Launch Streamlit App (1 minute)

```bash
streamlit run app.py
```

**The app opens at:** `http://localhost:8501`

---

## That's It! 🎉

Now you have:
✅ **3 Interactive Pages:**
1. 🎯 **Prediction Page** - Make real-time traffic forecasts
2. 📊 **Analytics Page** - Explore traffic patterns
3. ⭐ **Feature Importance** - Understand model decisions

---

## Common Commands

```bash
# Train models
python train.py

# Run Streamlit app
streamlit run app.py

# Make predictions from Python
python -c "from predict import predict_traffic_demand; print(predict_traffic_demand('qp02zt', 3, 'Residential', 3, 'Allowed', 'Yes', 28.5, 'Sunny', 14, 30))"

# Check installed packages
pip list | grep -E "streamlit|lightgbm|catboost|plotly"
```

---

## Troubleshooting

### Models not training?
```bash
# Verify data files exist
ls -la data/
# Should show: train.csv and test.csv
```

### App won't start?
```bash
# Try different port
streamlit run app.py --server.port 8502
```

### ImportError?
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## Next Steps

After training:
1. Explore the **Prediction Page** with different inputs
2. Check **Analytics Page** for traffic patterns
3. Review **Feature Importance** to understand model decisions
4. Modify hyperparameters in `train.py` for better results
5. Deploy to Streamlit Cloud (see README.md for instructions)

---

## Project Files Included

```
📦 Traffic-Demand-Forecasting/
├── 📄 app.py                  ← Run this for Streamlit app
├── 🔧 train.py               ← Run this to train models
├── 🎯 predict.py             ← Prediction logic
├── ⚙️ feature_engineering.py  ← Feature transformations
├── 🔌 model_loader.py        ← Model persistence
├── 🛠️ utils.py               ← Helper functions
├── 📋 requirements.txt        ← Python dependencies
├── 📚 README.md              ← Full documentation
├── 📂 data/
│   ├── train.csv            ← Your training data
│   └── test.csv             ← Your test data
└── 📂 models/               ← Models (created after train.py)
    ├── lightgbm_model.pkl
    ├── catboost_model.pkl
    └── feature_engineer.pkl
```

---

## Model Architecture

```
Input Features
    ↓
Feature Engineering (20+ features)
    ↓
    ├─ LightGBM Model (80% weight)
    │
    └─ CatBoost Model (20% weight)
    ↓
Ensemble Prediction (Weighted Average)
    ↓
Congestion Classification
    ↓
Output: Demand % + Congestion Level
```

---

## Production Deployment

Once you're satisfied with results, deploy to **Streamlit Cloud**:

1. Push code to GitHub
2. Go to `streamlit.io`
3. Click "New app"
4. Connect your GitHub repo
5. Deploy! 🚀

See README.md for detailed deployment steps.

---

**Happy Forecasting! 🚗** 

Questions? Check README.md for complete documentation.
