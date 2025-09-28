# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from newsapi import NewsApiClient

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="🚀 Real-Time News Emotion Dashboard", layout="wide")
st.title("🚀 Real-Time News Emotion Dashboard")
st.write("Predicting emotions of live news headlines using pre-trained Spark ML model converted to Python!")

# ---------------- Spark Session ----------------
spark = SparkSession.builder.appName("NewsEmotionApp").getOrCreate()

# ---------------- Load Model ----------------
model = PipelineModel.load("models/spark_linear_svc_pipeline")

# ---------------- NewsAPI ----------------
api_key = st.secrets["NEWSAPI_KEY"]
newsapi = NewsApiClient(api_key=api_key)

# ---------------- Sidebar Widgets ----------------
st.sidebar.title("Settings")

country = st.sidebar.selectbox(
    "Select Country", ["us", "gb", "in", "ca", "au"], key="country_select"
)
category = st.sidebar.selectbox(
    "Category", ["business", "entertainment", "general", "health", "science", "sports", "technology"],
    key="category_select"
)
page_size = st.sidebar.slider(
    "Number of Headlines", 5, 20, 10, key="page_size_slider"
)
refresh_rate = st.sidebar.number_input(
    "Auto-refresh every N seconds", min_value=60, max_value=3600, value=300, step=60, key="refresh_rate_input"
)

# Optional: Auto-refresh HTML meta
st.markdown(f'<meta http-equiv="refresh" content="{refresh_rate}">', unsafe_allow_html=True)

# ---------------- Fetch Headlines ----------------
@st.cache_data(ttl=refresh_rate)
def fetch_headlines(country, category, page_size):
    try:
        articles = newsapi.get_top_headlines(
            language='en', country=country, category=category, page_size=page_size
        )['articles']
        headlines = [a['title'] for a in articles if a['title']]
        return headlines
    except Exception as e:
        st.error(f"Error fetching headlines: {e}")
        return []

headlines = fetch_headlines(country, category, page_size)
if not headlines:
    st.warning("No headlines found. Try changing settings.")
    st.stop()

# ---------------- Convert to Spark DF ----------------
df = spark.createDataFrame([(h,) for h in headlines], ["headline"])

# ---------------- Predict Emotions ----------------
predictions = model.transform(df)
predictions = predictions.select("headline", "prediction").toPandas()

# Map numeric labels to emotion names
label_map = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
predictions['emotion'] = predictions['prediction'].apply(lambda x: label_map[int(x)])

# Store in session_state for continuity
st.session_state['predictions'] = predictions

# ---------------- Display ----------------
st.subheader("Live Headlines with Emotion Predictions")
st.dataframe(predictions[['headline', 'emotion']], use_container_width=True)

st.subheader("Emotion Distribution")
fig = px.pie(predictions, names='emotion', title="Emotion Distribution",
             color='emotion',
             color_discrete_map={
                 'joy': '#00CC96', 'sadness': '#636EFA', 'love': '#EF553B',
                 'anger': '#AB63FA', 'fear': '#FFA15A', 'surprise': '#19D3F3'
             })
st.plotly_chart(fig, use_container_width=True)

st.subheader("Bar Chart of Emotions")
st.bar_chart(predictions['emotion'].value_counts())

st.markdown("---")
st.info(f"Data from NewsAPI.org | Auto-refresh every {refresh_rate} seconds")
