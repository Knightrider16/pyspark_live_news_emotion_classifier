# app_sklearn_binary_real_time.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from newsapi import NewsApiClient
import re
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ------------------ Load scikit-learn binary model ------------------
data = joblib.load("models/sk_model_full.pkl")
model = data["model"]
num_features = data["num_features"]
stopwords = set(data["stopwords"])

# ------------------ Text preprocessing ------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stopwords]
    return ' '.join(tokens)

# ------------------ Streamlit page ------------------
st.set_page_config(page_title="Real-Time News Sentiment Dashboard", layout="wide")
st.title("Real-Time News Sentiment Dashboard")
st.markdown(
    "Predicting positive/negative sentiment of live news headlines using a pre-trained model!"
)

# ------------------ Sidebar ------------------
st.sidebar.title("Settings")
country = st.sidebar.selectbox("Select Country", ["us", "gb", "in", "ca", "au"])
category = st.sidebar.selectbox(
    "Select Category",
    ["business", "entertainment", "general", "health", "science", "sports", "technology"]
)
page_size = st.sidebar.slider("Number of Headlines", 5, 20, 10)
refresh_rate = st.sidebar.slider("Auto-refresh every (seconds)", 30, 300, 60)

# ------------------ Auto-refresh ------------------
count = st_autorefresh(interval=refresh_rate * 1000, limit=None, key="refresh")

# ------------------ Cached News Fetch Function ------------------
@st.cache_data(ttl=refresh_rate)
def fetch_news(country, category, page_size):
    api_key = st.secrets["NEWSAPI_KEY"]
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_top_headlines(
        language='en', country=country, category=category, page_size=page_size
    ).get('articles', [])
    headlines = [a['title'] for a in articles if a.get('title')]
    return headlines

# ------------------ Fetch Headlines ------------------
headlines = fetch_news(country, category, page_size)
if not headlines:
    st.warning("No headlines found. Try changing settings.")
    st.stop()

# ------------------ Preprocess ------------------
processed_headlines = [preprocess_text(h) for h in headlines]

# ------------------ Transform & Predict ------------------
vectorizer = HashingVectorizer(n_features=num_features, alternate_sign=False)
X = vectorizer.transform(processed_headlines)
preds = model.predict(X)

# ------------------ Map numeric labels to binary sentiment ------------------
label_map = {0: "negative", 1: "positive"}
sentiments = [label_map[int(p)] for p in preds]

# ------------------ Build DataFrame ------------------
df = pd.DataFrame({"headline": headlines, "sentiment": sentiments})

# ------------------ Display ------------------
st.subheader("Latest Headlines with Sentiment")
st.dataframe(df, use_container_width=True)

# ------------------ Sentiment Distribution ------------------
color_map = {"positive": "#00CC96", "negative": "#EF553B"}

# Pie chart
fig_pie = px.pie(df, names="sentiment", title="Sentiment Distribution", color="sentiment", color_discrete_map=color_map)
st.plotly_chart(fig_pie, use_container_width=True)

# Bar chart
counts = df["sentiment"].value_counts().reset_index()
counts.columns = ["sentiment", "count"]
fig_bar = px.bar(counts, x="sentiment", y="count", color="sentiment", color_discrete_map=color_map, title="Sentiment Counts")
st.plotly_chart(fig_bar, use_container_width=True)

st.info(f"Model: Pre-trained Spark binary sentiment model converted to Python | Auto-refresh every {refresh_rate} seconds")
