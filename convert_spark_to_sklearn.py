import joblib
from pyspark.ml.pipeline import PipelineModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer

# ------------------ Load Spark model ------------------
spark_model = PipelineModel.load("models/spark_linear_svc_pipeline")

# ------------------ Extract stages ------------------
tokenizer = spark_model.stages[0]
stopwords_remover = spark_model.stages[1]
hashingTF = spark_model.stages[2]
idf_model = spark_model.stages[3]
svc_model = spark_model.stages[4]

# ------------------ Save StopWords ------------------
stopwords_list = stopwords_remover.getStopWords()

# ------------------ Extract weights from LinearSVC ------------------
coefs = np.array(svc_model.coefficients).reshape(1, -1)
intercept = np.array([svc_model.intercept])

# ------------------ Create and configure LogisticRegression ------------------
sk_model = LogisticRegression()
sk_model.coef_ = coefs
sk_model.intercept_ = intercept
sk_model.classes_ = np.array([0, 1])

# Create a simple test to initialize the model properly
dummy_X = np.zeros((2, hashingTF.getNumFeatures()))
dummy_y = np.array([0, 1])
sk_model.fit(dummy_X, dummy_y)  # This properly initializes the model

# Now reassign the coefficients
sk_model.coef_ = coefs
sk_model.intercept_ = intercept

# ------------------ Save with minimal dependencies ------------------
model_data = {
    "model": sk_model,
    "num_features": hashingTF.getNumFeatures(),
    "stopwords": stopwords_list
}

# Save with protocol=4 for better compatibility
joblib.dump(model_data, "models/sk_model_full.pkl", protocol=4)

print("✅ Model converted and saved with scikit-learn compatibility!")