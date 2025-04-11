import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from ollama import Client
from groq import Client
from groq import Groq

# --- Districts and Mandals with coordinates ---
districts = {
    "Hyderabad": {
        "coordinates": (17.385044, 78.486671),
        "mandals": {
            "Amberpet": (17.3964, 78.5010),
            "Himayatnagar": (17.4000, 78.4800),
            "Nampally": (17.3850, 78.4747),
            "Asifnagar": (17.3833, 78.4500),
            "Saidabad": (17.3600, 78.5000)
        }
    },
    "Warangal": {
        "coordinates": (17.9784, 79.5941),
        "mandals": {
            "Chennaraopet": (17.9000, 79.7000),
            "Duggondi": (17.9500, 79.6500),
            "Narsampet": (17.9276, 79.8957),
            "Nekkonda": (17.8000, 79.7833),
            "Parvathagiri": (17.8500, 79.5500)
        }
    },
    "Nizamabad": {
        "coordinates": (18.6725, 78.0941),
        "mandals": {
            "Armoor": (18.7900, 78.2800),
            "Balkonda": (18.8500, 78.3200),
            "Bheemgal": (18.7300, 78.2900),
            "Bodhan": (18.6700, 77.9000),
            "Dharpalle": (18.6500, 78.2000)
        }
    },
    "Khammam": {
        "coordinates": (17.2473, 80.1514),
        "mandals": {
            "Khammam (Urban)": (17.2500, 80.1500),
            "Kothagudem": (17.5500, 80.6200),
            "Palwancha": (17.6000, 80.7000),
            "Yellandu": (17.6000, 80.3300),
            "Madhira": (17.2800, 80.3800)
        }
    },
    "Karimnagar": {
        "coordinates": (18.4386, 79.1288),
        "mandals": {
            "Karimnagar": (18.4386, 79.1288),
            "Kothapally": (18.4500, 79.1500),
            "Karimnagar Rural": (18.4200, 79.1000),
            "Manakondur": (18.4000, 79.2000),
            "Thimmapur": (18.3500, 79.1500)
        }
    },
    "Mahbubnagar": {
        "coordinates": (16.7471, 77.9854),
        "mandals": {
            "Addakal": (16.7000, 77.9833),
            "Balanagar": (16.9667, 78.1667),
            "Bhoothpur": (16.7167, 78.0000),
            "CC Kunta": (16.7500, 78.0500),
            "Devarakadra": (16.6500, 77.9000)
        }
    },
    "Adilabad": {
        "coordinates": (19.6669, 78.5320),
        "mandals": {
            "Adilabad": (19.6669, 78.5320),
            "Mancherial": (18.8800, 79.4400),
            "Nirmal": (19.1000, 78.3500),
            "Kagaznagar": (19.3333, 79.4667),
            "Mandamarri": (18.9667, 79.4667)
        }
    },
    "Rangareddy": {
        "coordinates": (17.3000, 78.2000),
        "mandals": {
            "Abdullapurmet": (17.3500, 78.6500),
            "Amangal": (16.8667, 78.5333),
            "Balapur": (17.3167, 78.5000),
            "Chevella": (17.3833, 78.1333),
            "Farooqnagar": (17.0833, 78.2000)
        }
    }
}

# --- Sidebar for location selection ---
st.sidebar.title("🌐 Select Your Location")
selected_district = st.sidebar.selectbox("District", list(districts.keys()))
mandal_options = districts[selected_district]["mandals"]
selected_mandal = st.sidebar.selectbox("Mandal", list(mandal_options.keys()))

# --- Weather forecast function ---
def forecast_temperature(lat, lon):
    session = requests_cache.CachedSession('.cache', expire_after=-1)
    session = retry(session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)

    end_date = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "windspeed_10m_max",
            "relative_humidity_2m_max"
        ],
        "timezone": "Asia/Kolkata"
    }

    response = client.weather_api(url, params=params)[0]
    daily = response.Daily()

    df = pd.DataFrame({
        "date": pd.to_datetime(daily.Time()),
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "precipitation": daily.Variables(2).ValuesAsNumpy(),
        "wind": daily.Variables(3).ValuesAsNumpy(),
        "humidity": daily.Variables(4).ValuesAsNumpy()
    }).dropna()

    features = df[["temp_max", "temp_min", "precipitation", "wind", "humidity"]].values
    n_steps = 25

    def preprocess_multivariate(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps][0])
        return np.array(X), np.array(y)

    X, y = preprocess_multivariate(features, n_steps)

    model = Sequential([
        GRU(64, activation='relu', input_shape=(n_steps, X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    model.fit(X, y, epochs=50, batch_size=20, verbose=0)

    def forecast_future(model, last_input, steps):
        preds = []
        current_input = last_input.copy()
        for _ in range(steps):
            pred = model.predict(current_input[np.newaxis, :, :], verbose=0)
            preds.append(pred[0][0])
            next_row = current_input[-1].copy()
            next_row[0] = pred[0][0]
            current_input = np.vstack([current_input[1:], next_row])
        return np.array(preds)

    last_input = features[-n_steps:]
    return forecast_future(model, last_input, 7)

# --- Get location ---
lat, lon = mandal_options[selected_mandal]
forecasted_temps = forecast_temperature(lat, lon)

# --- Use fixed Monday-to-Sunday list ---
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- Plot the forecast ---
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=days,
    y=forecasted_temps,
    mode='lines+markers',
    name='Max Temp Forecast',
    line=dict(color='orangered', width=3),
    marker=dict(size=8)
))
fig.update_layout(
    title=f"📊 7-Day Max Temperature Forecast for {selected_mandal}, {selected_district}",
    xaxis_title="Day",
    yaxis_title="Temperature (°C)",
    template='plotly_white'
)
st.plotly_chart(fig)

# --- Generate full emoji/mood descriptions using Groq ---
def get_llm_daywise_descriptions(district, mandal, temps, days):
    client = Client(api_key="your_GROQCLOUD_APIKEY")  # Replace with your key
    forecast_data = "\n".join([f"{day}: {round(temp,1)}°C" for day, temp in zip(days, temps)])
    prompt = f"""
You are a cheerful and smart weather assistant. Here's the 7-day forecast for {mandal}, {district}:

{forecast_data}

For each day, give a mood/emoji + 1-line friendly description of the weather and any helpful advice.
Output it in this format:
Day - Mood + Tip

Example:
Monday - 🔥 Scorching heat! Avoid afternoon sun and hydrate well.
Tuesday - 🌞 Bright and sunny! Great for laundry or a morning walk.
"""
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",  # or "llama3-8b-8192"
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


import streamlit as st
from groq import Groq

# --- Generate summary using Groq ---
def get_llm_summary(district, mandal, temps, days):
    client = Groq(api_key="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")  # Replace with your key
    forecast_data = "\n".join([f"{day}: {round(temp,1)}°C" for day, temp in zip(days, temps)])
    prompt = f"""
You are a friendly weather assistant. Here's the 7-day max temperature forecast for {mandal}, {district}:

{forecast_data}

Generate a natural language summary explaining the weekly trend, weather comfort, and public advice.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# --- Display table ---
st.markdown("### 📅 Daily Forecast Summary (Powered by LLM 🧠)")
daywise_descriptions = get_llm_summary(selected_district, selected_mandal, forecasted_temps, days)
st.code(daywise_descriptions)

# --- Display Weekly Summary ---
st.markdown("### 🗒 Weekly Weather Summary")
summary = get_llm_summary(selected_district, selected_mandal, forecasted_temps, days)
st.write(summary)

# --- Disclaimer ---
st.markdown("""
---
📌 Note: This is an AI-based weather forecasting model. While it uses historical data and modern techniques, it may not always be accurate.
""")
