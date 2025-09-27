# app.py
import streamlit as st
from newsapi import NewsApiClient
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pandas as pd

# ---------- Streamlit page ----------
st.set_page_config(page_title="News Emotion Dashboard", layout="wide")
st.title("Real-Time News Emotion Classification")
st.write("Classifies live news headlines into 6 emotions: sadness, joy, love, anger, fear, surprise.")

# ---------- Spark session ----------
spark = SparkSession.builder.appName("NewsEmotionApp").getOrCreate()

# ---------- Load trained PySpark ML pipeline ----------
model = PipelineModel.load("spark_linear_svc_pipeline")

# ---------- NewsAPI ----------
api_key = st.secrets["NEWSAPI_KEY"]
newsapi = NewsApiClient(api_key=api_key)

# ---------- Sidebar Settings ----------
st.sidebar.title("Settings")
country = st.sidebar.selectbox("Select Country", ["us", "gb", "in", "ca", "au"])
category = st.sidebar.selectbox("Category", ["business", "entertainment", "general", "health", "science", "sports", "technology"])
page_size = st.sidebar.slider("Number of Headlines", 5, 20, 10)

# ---------- Auto-refresh every 5 minutes ----------
st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)

# ---------- Fetch news (cached for 5 min) ----------
@st.cache_data(ttl=300)
def fetch_news():
    articles = newsapi.get_top_headlines(
        language='en', country=country, category=category, page_size=page_size
    )['articles']
    headlines = [a['title'] for a in articles if a['title']]
    return headlines

headlines = fetch_news()
if not headlines:
    st.warning("No headlines found. Try changing settings.")
    st.stop()

# ---------- Convert to Spark DataFrame ----------
df = spark.createDataFrame([(h,) for h in headlines], ["headline"])

# ---------- Predict ----------
predictions = model.transform(df)
predictions = predictions.select("headline", "prediction").toPandas()

# ---------- Map numeric labels to emotion names ----------
label_map = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
predictions['emotion'] = predictions['prediction'].apply(lambda x: label_map[int(x)])

# ---------- Display ----------
st.subheader("Live Headlines with Emotion Predictions")
st.dataframe(predictions[['headline', 'emotion']])

st.subheader("Emotion Distribution")
st.bar_chart(predictions['emotion'].value_counts())
