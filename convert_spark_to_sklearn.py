# convert_spark_to_sklearn_binary.py
import joblib
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
import numpy as np
from sklearn.linear_model import LogisticRegression

# ------------------ Load Spark model ------------------
spark_model = PipelineModel.load("models/spark_linear_svc_pipeline")

# ------------------ Extract stages ------------------
tokenizer = spark_model.stages[0]       # Tokenizer
stopwords_remover = spark_model.stages[1]  # StopWordsRemover
hashingTF = spark_model.stages[2]       # HashingTF
idf_model = spark_model.stages[3]       # IDF
svc_model = spark_model.stages[4]       # LinearSVC (binary)

# ------------------ Save StopWords ------------------
stopwords_list = stopwords_remover.getStopWords()

# ------------------ Extract weights from LinearSVC ------------------
coefs = np.array(svc_model.coefficients).reshape(1, -1)
intercept = np.array([svc_model.intercept])

# ------------------ Create dummy LogisticRegression ------------------
# Needs at least 2 classes for sklearn
dummy_X = np.zeros((2, hashingTF.getNumFeatures()))
dummy_y = np.array([0, 1])
sk_model = LogisticRegression()
sk_model.fit(dummy_X, dummy_y)

# ------------------ Assign coefficients ------------------
sk_model.coef_ = coefs
sk_model.intercept_ = intercept
sk_model.classes_ = np.array([0, 1])

# ------------------ Save everything needed for Python inference ------------------
joblib.dump({
    "model": sk_model,
    "num_features": hashingTF.getNumFeatures(),
    "stopwords": stopwords_list
}, "models/sk_model_full.pkl")

print("✅ Spark binary model converted to scikit-learn with preprocessing!")
