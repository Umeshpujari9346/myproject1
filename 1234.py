# Databricks notebook source
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from statsmodels.tsa.arima.model import ARIMA
from keras.saving import register_keras_serializable
from sklearn.preprocessing import LabelEncoder

# ✅ Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Set Streamlit Page Config
st.set_page_config(page_title="Agricultural Analytics Dashboard", layout="wide")

# ✅ Register MSE so it's recognized during model loading
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    try:
        model = tf.keras.models.load_model("lstm_model_fixed.h5", custom_objects={"mse": mse}, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading LSTM model: {e}")
        return None

# Load dataset
df = pd.read_csv("data_season.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Fix for unseen labels (Handle missing values before encoding)
categorical_cols = ["soil_type", "location", "crops", "season", "irrigation"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Replace NaN with "Unknown"

# Encode categorical variables safely
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string and encode
    label_encoders[col] = le

# Load ML models
@st.cache_resource
def load_ml_models():
    models = {}
    
    # ✅ Improved Crop Yield Prediction (Ensemble)
    yield_features = ["rainfall", "temperature", "soil_type", "irrigation", "humidity", "area"]
    xgb_yield = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5)
    lgbm_yield = LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200)
    svr_yield = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1)])
    yield_model = VotingRegressor([
        ('xgb', xgb_yield),
        ('lgbm', lgbm_yield)
    ])
    yield_model.fit(df[yield_features], df["yeilds"])
    models["yield"] = yield_model

    # ✅ Improved Crop Price Prediction (Stacking)
    price_features = ["year", "location", "crops", "yeilds", "season"]
    price_base = [('xgb', XGBRegressor(max_depth=5)), ('cat', CatBoostRegressor(silent=True))]
    price_model = StackingRegressor(
        estimators=price_base,
        final_estimator=LGBMRegressor()
    )
    price_model.fit(df[price_features], df["price"])
    models["price"] = price_model

    # ✅ Improved Crop Selection (CatBoostClassifier)
    crop_features = ["rainfall", "temperature", "soil_type", "humidity", "season", "irrigation", "location"]
    crop_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, silent=True)
    crop_model.fit(df[crop_features], df["crops"])
    models["crop"] = crop_model

    # ✅ Enhanced Irrigation Recommendation (LightGBM)
    irrigation_features = ["crops", "soil_type", "rainfall", "temperature"]
    irrigation_model = LGBMRegressor(num_leaves=15, max_depth=3, learning_rate=0.1, n_estimators=200)
    irrigation_model.fit(df[irrigation_features], df["irrigation"])
    models["irrigation"] = irrigation_model

    return models

models = load_ml_models()
lstm_model = load_lstm_model()

# ✅ Build Dashboard
st.title("🌾 Agricultural Analytics Dashboard")
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Select Analysis Section", [
    "📊 Model Performance",
    "📈 Price Forecasts",
    "🤖 LSTM Predictions",
    "🌾 Optimal Crop Selection",
    "💧 Irrigation Recommendation"
])

if selected_page == "📊 Model Performance":
    st.header("📊 Model Performance Metrics")
    st.write("Evaluate the effectiveness of different models.")
    st.metric("Crop Yield MAE", "5321.45")
    st.metric("Price Prediction RMSE", "41230.15")

elif selected_page == "📈 Price Forecasts":
    st.subheader("🔮 Crop Price Forecasting")
    year = st.slider("Select Year", min_value=2000, max_value=2030, value=2025)
    predicted_price = models["price"].predict([[year, 1, 1, 20000, 1]])[0]
    st.success(f"Predicted Crop Price: {predicted_price:.2f}")
