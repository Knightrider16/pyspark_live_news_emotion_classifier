# app_sklearn_full_real_time.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from newsapi import NewsApiClient
import re
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------- Load scikit-learn model ----------
data = joblib.load("models/sk_model_full.pkl")
model = data["model"]
num_features = data["num_features"]
stopwords = set(data["stopwords"])

# ---------- Text preprocessing ----------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in stopwords]
    return ' '.join(tokens)

# ---------- Streamlit page ----------
st.set_page_config(page_title="Real-Time News Emotion Dashboard", layout="wide")
st.title("🚀 Real-Time News Emotion Dashboard")
st.markdown("Predicting emotions of live news headlines using pre-trained Spark ML model converted to Python!")

# ---------- Sidebar ----------
st.sidebar.title("Settings")
country = st.sidebar.selectbox("Select Country", ["us", "gb", "in", "ca", "au"])
category = st.sidebar.selectbox("Category", ["business", "entertainment", "general", "health", "science", "sports", "technology"])
page_size = st.sidebar.slider("Number of Headlines", 5, 20, 10)
refresh_rate = st.sidebar.slider("Auto-refresh every (seconds)", 30, 300, 60)

# ---------- Auto-refresh ----------
count = st_autorefresh(interval=refresh_rate*1000, limit=None, key="refresh")
st.markdown(f'<meta http-equiv="refresh" content="{refresh_rate}">', unsafe_allow_html=True)

# ---------- Fetch news ----------
api_key = st.secrets["NEWSAPI_KEY"]
newsapi = NewsApiClient(api_key=api_key)

@st.cache_data(ttl=refresh_rate)
def fetch_news():
    try:
        articles = newsapi.get_top_headlines(language='en', country=country, category=category, page_size=page_size)['articles']
        headlines = [a['title'] for a in articles if a['title']]
        return headlines
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

headlines = fetch_news()
if not headlines:
    st.warning("No headlines found. Try changing settings.")
    st.stop()

# ---------- Preprocess ----------
processed_headlines = [preprocess_text(h) for h in headlines]

# ---------- Transform & Predict ----------
vectorizer = HashingVectorizer(n_features=num_features, alternate_sign=False)
X = vectorizer.transform(processed_headlines)
preds = model.predict(X)

# ---------- Map numeric labels ----------
label_map = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
emotions = [label_map[int(p)] for p in preds]

# ---------- Build DataFrame ----------
df = pd.DataFrame({"headline": headlines, "emotion": emotions})

# ---------- Display ----------
st.subheader("Latest Headlines with Emotions")
st.dataframe(df, use_container_width=True)

# ---------- Emotion Distribution ----------
st.subheader("Emotion Distribution")
color_map = {
    'joy': '#00CC96',
    'sadness': '#636EFA',
    'love': '#AB63FA',
    'anger': '#EF553B',
    'fear': '#FFA15A',
    'surprise': '#19D3F3'
}

# Pie chart
fig_pie = px.pie(df, names='emotion', title="Emotion Distribution", color='emotion', color_discrete_map=color_map)
st.plotly_chart(fig_pie, use_container_width=True)

# Bar chart
st.subheader("Emotion Counts")
counts = df['emotion'].value_counts().reset_index()
counts.columns = ['emotion', 'count']
fig_bar = px.bar(counts, x='emotion', y='count', color='emotion', color_discrete_map=color_map,
                 title="Emotion Counts")
st.plotly_chart(fig_bar, use_container_width=True)

st.info(f"Model: Pre-trained Spark ML converted to Python | Auto-refresh every {refresh_rate} seconds")
